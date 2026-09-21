from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import httpx
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
)
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn

logger = logging.getLogger(__name__)

DATA_PREFIX = "data:"
EVENT_PREFIX = "event:"
UNUSED_API_KEY = "none"

# Mirrors FlueLanguageModelHandlerArguments.flue_request_timeout_s's default: the
# registry always supplies a value, so this only matters for direct construction
# (tests). setup() below injects this same value into kwargs before delegating to
# super().setup(), so the base class's self.request_timeout_s and this handler's
# own httpx.Client read timeout never see two different defaults.
DEFAULT_REQUEST_TIMEOUT_S = 300.0

# Best-effort and short on purpose: this fires during cleanup for a turn we're
# already giving up on, and must never make that cleanup itself slow.
ABORT_TIMEOUT_S = 3.0


def parse_flue_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield each chunk object from every complete ``event: data`` SSE frame in *lines*.

    flue's ``data:`` payload is a JSON array of chunks per frame, and every frame
    carries an ``event:`` line (``data`` or ``control``). Only ``data`` frames carry
    conversation chunks; ``control`` frames carry stream bookkeeping
    (``streamNextOffset``/``upToDate``) and are skipped. A frame not terminated by a
    blank line (the underlying stream was cut off mid-frame) is dropped, never
    parsed: the caller tells "ended cleanly" from "cut off" by whether a terminal
    event was seen, not by whether parsing raised.
    """
    event_type: Optional[str] = None
    data: list[str] = []

    for line in lines:
        if line.startswith(EVENT_PREFIX):
            event_type = line[len(EVENT_PREFIX) :].strip()
        elif line.startswith(DATA_PREFIX):
            data.append(line[len(DATA_PREFIX) :].lstrip())
        elif not line.strip():
            if event_type == "data" and data:
                yield from json.loads("\n".join(data))
            event_type = None
            data = []


def settlement_from_snapshot(snapshot: dict[str, Any], submission_id: str) -> Optional[dict[str, Any]]:
    """Find *submission_id*'s own settlement inside a ``conversation-reset`` snapshot.

    flue's auto-compaction can replace the batch holding a turn's settlement with a
    full ``conversation-reset`` chunk; when that happens the settlement exists only
    in the snapshot's ``settlements``, never as its own ``submission-settled`` chunk.

    A submission's own settlement record always carries its own id as
    ``submissionId`` — including a delivery that joined another response, whose
    record adds ``answeredBySubmissionId`` pointing at the host it joined.
    Matching only on ``submissionId`` (not also trying ``answeredBySubmissionId``)
    means this never mistakes a *later* submission that joined onto *us* (when we
    are the host) for our own settlement.
    """
    for settlement in snapshot.get("settlements", []):
        if settlement.get("submissionId") == submission_id:
            return settlement
    return None


def message_text(content: Any) -> str:
    """Flatten a flue message content, which is a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class FlueModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler that drives a flue agent through its per-conversation HTTP+SSE API."""

    def setup(
        self,
        *args: Any,
        agent_name: Optional[str] = None,
        session_gap_hours: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._session_gap_s = session_gap_hours * 3600
        self._conversation_id: Optional[str] = None
        self._last_turn_at: Optional[float] = None
        base_url = kwargs.get("base_url")
        # setdefault (not get): the same value must also reach super().setup()
        # below, which is what actually sets self.request_timeout_s — otherwise
        # this default and the base class's own (unrelated) default could each
        # apply to a different half of this handler.
        request_timeout_s = kwargs.setdefault("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        if not base_url or not agent_name:
            raise ValueError("The flue backend requires --flue_base_url and --flue_agent_name.")
        self.flue_url = base_url.rstrip("/")
        self.agent_name = agent_name
        # flue's SSE keeps the connection alive with a heartbeat every 15s, even
        # while a tool runs, so httpx's own read timeout never fires on its own
        # (see _deadline_lines). A short, separate connect timeout still catches an
        # unreachable host quickly.
        self.http = httpx.Client(
            base_url=self.flue_url,
            timeout=httpx.Timeout(request_timeout_s, connect=min(10.0, request_timeout_s)),
        )
        kwargs["api_key"] = UNUSED_API_KEY
        try:
            super().setup(*args, **kwargs)
        except BaseException:
            # super().setup() runs warmup() and (if misconfigured) raises before
            # this handler is otherwise usable — nothing will call cleanup() on a
            # construction that never finished, so this client would otherwise
            # leak its connection.
            self.http.close()
            raise
        logger.debug("flue backend ready: url=%s agent=%s", self.flue_url, self.agent_name)

    def cleanup(self) -> None:
        self.http.close()

    def warmup(self) -> None:
        # flue has no agent-catalog endpoint: there is nothing to resolve or
        # validate the agent name against ahead of time. This only checks that the
        # flue server itself is reachable; an unknown agent name still only fails
        # on the first real turn (as a 404 from its conversation route).
        try:
            self.http.get("/")
        except httpx.TransportError as exc:
            raise ValueError(f"flue server unreachable: {self.flue_url}") from exc

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        raise ValueError("The flue backend owns its history; run with --compact_history false.")

    def _serialize(self, active_chat: Chat) -> str:
        for message in reversed(active_chat.to_transformers_chat()):
            if message.get("role") == "user":
                return message_text(message.get("content"))
        return ""

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        return {}

    def _current_conversation_id(self) -> str:
        """Return the conversation id for this turn, minting a new one if idle too long.

        A new id is minted on the first turn, and again whenever at least
        ``self._session_gap_s`` has passed since the previous turn (0 means every
        turn). ``self._last_turn_at`` always advances to "now" so the gap is measured
        from the most recent turn, not from when the current conversation id started.
        """
        now = time.time()
        if self._conversation_id is None or now - self._last_turn_at >= self._session_gap_s:
            self._conversation_id = str(uuid.uuid4())
        self._last_turn_at = now
        return self._conversation_id

    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
        # A plain generator, unlike the openai.Stream other backends return here,
        # cannot be closed safely from a different thread while it is actively
        # running (Python raises "generator already executing"). The base class
        # only ever attempts that for its tool-call prefetch worker, which never
        # engages for this backend: _build_optional_kwargs never sends tools, so
        # _iter_stream_events never yields a ToolCall, so the prefetch trigger
        # (a completed tool call) can never fire. If this backend ever gains tool
        # support, that assumption stops holding and this needs revisiting.
        return self._stream_turn(self._current_conversation_id(), api_input)

    def _iter_stream_events(self, api_response: Iterator[dict[str, Any]]) -> Iterator[ProviderEvent]:
        # _stream_turn locks onto one message id at a time, but can still switch
        # which one mid-turn (a stale leftover it gives up on, then its own real
        # message) without a message-completed in between — the abandoned one
        # simply stops being forwarded. Tracking the id here too means a switch
        # discards that stale text instead of silently gluing it onto the next
        # message as if they were one continuous reply.
        buffer = ""
        current_message_id: Optional[str] = None
        for event in api_response:
            event_type = event.get("type")
            if event_type == "message-delta" and event.get("kind") == "text":
                message_id = event.get("messageId")
                if message_id != current_message_id:
                    buffer = ""
                    current_message_id = message_id
                delta = event.get("delta") or ""
                if delta:
                    buffer += delta
                    yield TextDelta(text=delta)
            elif event_type == "message-completed":
                if event.get("messageId") == current_message_id and buffer:
                    yield AssistantMessage(content=[AssistantContent(type="output_text", text=buffer)])
                buffer = ""
                current_message_id = None
            elif event_type == "submission-settled":
                outcome = event.get("outcome")
                if outcome == "completed":
                    if buffer:
                        yield AssistantMessage(content=[AssistantContent(type="output_text", text=buffer)])
                elif outcome != "aborted":
                    raise RuntimeError(f"flue submission {outcome}: {event.get('error')}")
                # "aborted" only ever comes from our own _abort_submission or a
                # barge-in whose abort call drags this turn down too (flue's abort
                # targets the whole conversation instance, not just one
                # submission — see _abort_submission) — never a genuine backend
                # failure — so it ends the turn quietly instead of surfacing as
                # PROVIDER_FAILURE_FALLBACK.
                buffer = ""

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        raise ValueError("The flue backend always streams conversation events.")

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def _deadline_lines(self, lines: Iterable[str], deadline: float) -> Iterator[str]:
        """Enforce ``request_timeout_s`` as a wall-clock deadline on the whole turn.

        flue's SSE sends a heartbeat every 15s regardless of tool activity, which
        keeps httpx's own read timeout from ever firing on a turn that never
        settles. This raises the same ``httpx.ReadTimeout`` the base class already
        handles, so a stuck turn still ends instead of hanging the handler thread
        forever. *deadline* is an absolute `time.monotonic()` value set once for the
        whole turn (POST included), not restarted when the SSE read begins.
        """
        for line in lines:
            if time.monotonic() > deadline:
                raise httpx.ReadTimeout(
                    f"flue conversation stream exceeded {self.request_timeout_s:.0f}s without settling"
                )
            yield line

    def _abort_submission(self, path: str) -> None:
        """Best-effort: tell flue to stop generating a turn this adapter gave up on.

        flue has no way to learn that on its own — closing our side of the SSE read
        doesn't tell the server anything, so an abandoned turn (deadline, barge-in,
        a request error) would otherwise keep the model generating a response
        nobody will use, and keep it sitting in the conversation as a leftover this
        adapter's next turn then has to filter back out. This is fire-and-forget:
        the abort itself settles asynchronously server-side, and a failure here
        must never mask whatever error caused the turn to be abandoned in the
        first place.
        """
        try:
            self.http.post(f"{path}/abort", timeout=ABORT_TIMEOUT_S)
        except Exception as exc:
            # Anything here (not just httpx.HTTPError — e.g. posting to an
            # already-closed client) must be swallowed the same way: this runs
            # from a `finally`, so letting it escape would replace whatever error
            # actually caused the turn to be abandoned in the first place.
            logger.warning("flue abort request failed for %s: %s", path, exc)

    def _stream_turn(self, conversation_id: str, text: str) -> Iterator[dict[str, Any]]:
        deadline = time.monotonic() + self.request_timeout_s
        path = f"/agents/{self.agent_name}/{conversation_id}"
        admission = self._json("POST", path, json={"kind": "user", "body": text})
        submission_id = admission["submissionId"]
        # The message that answers this submission may already have been generating
        # before this delivery joined it (flue folds a delivery arriving mid-response
        # into that response instead of starting a fresh one — see
        # agent-submissions.ts's join handling), in which case its own
        # `message-started` chunk lies before this admission's offset and is never
        # seen here at all. So the first message-bearing chunk seen is provisionally
        # adopted as ours even before we know that — but a `message-started` that
        # DOES carry our own submissionId always overrides it (a fresh response of
        # ours is unambiguous), and a settlement that turns out to belong to someone
        # else clears the provisional id again. Both guard against locking onto a
        # leftover response from an earlier turn this adapter already gave up on
        # that is still trickling in when this read begins. The turn itself only
        # ends on a settlement matched on submissionId: a submission's own
        # settlement record always carries its own id there, whether it settled
        # directly or (having joined another response) via answeredBySubmissionId
        # pointing at the host — so submissionId alone is enough, and doesn't risk
        # a later, unrelated submission that joins onto *us* ending this turn early.
        message_id: Optional[str] = None
        turn_settled = False
        try:
            with self.http.stream(
                "GET",
                path,
                params={"view": "updates", "offset": admission["offset"], "live": "sse"},
                headers={"accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                for chunk in parse_flue_sse_events(self._deadline_lines(response.iter_lines(), deadline)):
                    chunk_type = chunk.get("type")
                    if chunk_type == "message-started":
                        if chunk.get("submissionId") == submission_id:
                            # A fresh response of our own always carries our own
                            # submissionId here, and a joined delivery's never does —
                            # so this always wins over whatever (if anything) was
                            # provisionally locked in below from an unrelated
                            # leftover response still trickling in from a turn this
                            # adapter already gave up on. A joined delivery's own
                            # message-started (submissionId != ours) needs no
                            # separate handling here: the message-delta/-completed
                            # branch below provisionally adopts the same id off its
                            # first chunk regardless of which event carried it first.
                            message_id = chunk.get("messageId")
                    elif chunk_type in ("message-delta", "message-completed"):
                        candidate = chunk.get("messageId")
                        if message_id is None:
                            message_id = candidate
                        if candidate == message_id:
                            yield chunk
                    elif chunk_type in ("submission-settled", "conversation-reset"):
                        # flue delivers "a submission settled" two ways: directly, or
                        # (after auto-compaction) folded into a conversation-reset's
                        # snapshot. Either way, a settlement that isn't ours confirms
                        # whatever message id we were tracking belonged to that other,
                        # now-finished submission: stop accepting its content and wait
                        # for our own.
                        if chunk_type == "conversation-reset":
                            settled = settlement_from_snapshot(chunk.get("snapshot") or {}, submission_id)
                            settled = {"type": "submission-settled", **settled} if settled is not None else None
                        elif chunk.get("submissionId") == submission_id:
                            settled = chunk
                        else:
                            settled = None
                        if settled is not None:
                            turn_settled = True
                            yield settled
                            return
                        message_id = None
            raise RuntimeError("flue conversation stream ended before the submission settled")
        finally:
            if not turn_settled:
                self._abort_submission(path)
