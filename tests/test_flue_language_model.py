import json
import queue
import threading
from datetime import datetime, timezone

import httpx
import pytest
from openai.types.realtime.conversation_item import RealtimeConversationItemAssistantMessage
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)

from speech_to_speech.arguments_classes.flue_language_model_arguments import FlueLanguageModelHandlerArguments
from speech_to_speech.LLM import flue_language_model as flue_mod
from speech_to_speech.LLM.base_openai_compatible_language_model import AssistantMessage, TextDelta
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.flue_language_model import (
    FlueModelHandler,
    conversation_id_for,
    parse_flue_sse_events,
    settlement_from_snapshot,
)


def _sse(*batches):
    return "".join(f"event: data\ndata:{json.dumps(batch)}\n\n" for batch in batches)


def _make_handler(transport):
    handler = object.__new__(FlueModelHandler)
    handler.flue_url = "http://flue"
    handler.agent_name = "flue-voice"
    handler.request_timeout_s = 300.0
    handler.http = httpx.Client(base_url=handler.flue_url, transport=transport)
    return handler


def _make_real_handler(transport, **extra_setup_kwargs):
    """Build a handler through the real BaseHandler.__init__ + setup() path, with the
    exact kwarg names backend_registry's config_prefix normalization produces. This
    would catch a renamed dataclass field or config_prefix that the mock-based
    _make_handler() above cannot: that one bypasses setup() entirely.

    The base class's own setup() calls warmup() before returning, so the mock
    transport must already be wired into the httpx.Client setup() constructs —
    swapping handler.http afterwards would be too late.

    flue_language_model.py does a plain ``import httpx``, so patching its
    ``Client`` patches the same global module object the OpenAI SDK's own client
    construction (inside the base class's setup()) would otherwise also see.
    FlueModelHandler.setup() only ever calls ``httpx.Client(...)`` once, for its
    own ``self.http``, strictly before delegating to the base class — so
    ``fake_client`` restores the real constructor the instant it is called,
    keeping the patched window to that one call instead of the whole construction.
    """
    real_client = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        flue_mod.httpx.Client = real_client
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    flue_mod.httpx.Client = fake_client
    try:
        setup_kwargs = dict(
            base_url="http://flue",
            agent_name="flue-voice",
            request_timeout_s=5.0,
            compact_history=False,
        )
        setup_kwargs.update(extra_setup_kwargs)
        handler = FlueModelHandler(
            threading.Event(),
            queue_in=queue.Queue(),
            queue_out=queue.Queue(),
            setup_kwargs=setup_kwargs,
        )
    finally:
        flue_mod.httpx.Client = real_client
    return handler


def test_setup_wires_attributes_from_registry_style_kwargs():
    handler = _make_real_handler(httpx.MockTransport(lambda request: httpx.Response(500)))

    assert handler.flue_url == "http://flue"
    assert handler.agent_name == "flue-voice"
    assert handler.request_timeout_s == 5.0


def test_setup_gives_the_http_client_a_short_connect_timeout():
    # Matches the base class's own convention (connect=min(10, request_timeout_s))
    # so an unreachable host fails fast at warmup instead of waiting the full
    # request_timeout_s just to establish a TCP connection. request_timeout_s must
    # exceed 10 here, or connect and read collapse to the same value and this
    # assertion would still pass even if the connect=min(10, ...) cap were removed
    # entirely.
    handler = _make_real_handler(httpx.MockTransport(lambda request: httpx.Response(500)), request_timeout_s=300.0)

    assert handler.http.timeout.connect == 10.0
    assert handler.http.timeout.read == 300.0


def test_setup_syncs_the_default_request_timeout_with_the_base_class():
    # Without an explicit request_timeout_s, the value used for the httpx.Client's
    # own read timeout and the value the base class stores as self.request_timeout_s
    # (the turn deadline) must be the same constant, not two independently-chosen
    # defaults that happen to both apply to different halves of this handler.
    real_client = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        flue_mod.httpx.Client = real_client
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(500))
        return real_client(*args, **kwargs)

    flue_mod.httpx.Client = fake_client
    try:
        handler = FlueModelHandler(
            threading.Event(),
            queue_in=queue.Queue(),
            queue_out=queue.Queue(),
            setup_kwargs=dict(base_url="http://flue", agent_name="flue-voice", compact_history=False),
        )
    finally:
        flue_mod.httpx.Client = real_client

    assert handler.request_timeout_s == flue_mod.DEFAULT_REQUEST_TIMEOUT_S
    assert handler.http.timeout.read == flue_mod.DEFAULT_REQUEST_TIMEOUT_S


def test_compact_history_true_is_rejected_at_setup_through_the_real_constructor():
    # Exercises the actual wiring: base_openai_compatible_language_model.setup()
    # calls _build_compaction_generate_fn() eagerly when compact_history=True, so
    # this must fail at construction, not just when the method is called directly.
    with pytest.raises(ValueError, match="owns its history"):
        _make_real_handler(httpx.MockTransport(lambda request: httpx.Response(500)), compact_history=True)


def test_setup_closes_the_http_client_if_super_setup_fails():
    # super().setup() runs warmup() and (for compact_history=True) rejects the
    # config after this handler's own http client already exists. Nothing will
    # ever call cleanup() on a construction that never finished, so setup() must
    # close it itself instead of leaking the connection.
    created_clients = []
    real_client = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        flue_mod.httpx.Client = real_client
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(500))
        client = real_client(*args, **kwargs)
        created_clients.append(client)
        return client

    flue_mod.httpx.Client = fake_client
    try:
        with pytest.raises(ValueError, match="owns its history"):
            FlueModelHandler(
                threading.Event(),
                queue_in=queue.Queue(),
                queue_out=queue.Queue(),
                setup_kwargs=dict(base_url="http://flue", agent_name="flue-voice", compact_history=True),
            )
    finally:
        flue_mod.httpx.Client = real_client

    assert len(created_clients) == 1
    assert created_clients[0].is_closed


def test_warmup_raises_when_the_flue_server_is_unreachable():
    def respond(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ValueError, match="flue server unreachable"):
        _make_real_handler(httpx.MockTransport(respond))


def test_setup_requires_base_url_and_agent_name():
    handler = object.__new__(FlueModelHandler)
    with pytest.raises(ValueError, match="flue_base_url"):
        FlueModelHandler.setup(handler, agent_name=None, base_url=None)


def test_setup_requires_base_url_even_when_agent_name_is_given():
    handler = object.__new__(FlueModelHandler)
    with pytest.raises(ValueError, match="flue_base_url"):
        FlueModelHandler.setup(handler, agent_name="flue-voice", base_url=None)


def test_setup_requires_agent_name_even_when_base_url_is_given():
    handler = object.__new__(FlueModelHandler)
    with pytest.raises(ValueError, match="flue_agent_name"):
        FlueModelHandler.setup(handler, agent_name=None, base_url="http://flue")


def test_compact_history_defaults_to_false():
    # base_openai_compatible_language_model's own default is True; this field
    # overrides it. If the override were ever dropped, every flue user's startup
    # would fail at construction (compact_history=True is rejected below).
    assert FlueLanguageModelHandlerArguments().compact_history is False


def test_compaction_is_rejected():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    with pytest.raises(ValueError, match="owns its history"):
        handler._build_compaction_generate_fn()


def test_cleanup_closes_the_http_client():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    handler.cleanup()
    assert handler.http.is_closed


def test_conversation_id_for_uses_the_jst_calendar_day():
    # 2026-09-18T15:00:00Z is exactly 2026-09-19T00:00 JST: the rollover instant.
    before_midnight = datetime(2026, 9, 18, 14, 59, 59, tzinfo=timezone.utc)
    at_midnight = datetime(2026, 9, 18, 15, 0, 0, tzinfo=timezone.utc)

    assert conversation_id_for(before_midnight) == "session20260918"
    assert conversation_id_for(at_midnight) == "session20260919"


def test_parse_flue_sse_events_reads_complete_data_frames_only():
    lines = [
        "event: data",
        'data:[{"type": "stream-checkpoint", "incarnation": "inc_1"}]',
        "",
        "event: control",
        'data:{"streamNextOffset": "1", "upToDate": true}',
        "",
        "event: data",
        'data:[{"type": "message-delta", "kind": "text",',
        'data:"delta": "hi"}]',
        "",
    ]

    assert list(parse_flue_sse_events(lines)) == [
        {"type": "stream-checkpoint", "incarnation": "inc_1"},
        {"type": "message-delta", "kind": "text", "delta": "hi"},
    ]


def test_parse_flue_sse_events_drops_an_unterminated_trailing_frame():
    # No trailing blank line: the connection was cut off mid-frame. The frame must
    # be dropped, not parsed (a truncated JSON array would raise anyway, but an
    # otherwise-complete-looking payload without its terminator must still be
    # treated as "never happened", not "happened".
    lines = [
        "event: data",
        'data:[{"type": "message-delta", "kind": "text", "delta": "ok"}]',
        "",
        "event: data",
        'data:[{"type": "message-delta", "kind": "text", "delta": "cut off"}]',
        # stream ends here, no blank line
    ]

    assert list(parse_flue_sse_events(lines)) == [
        {"type": "message-delta", "kind": "text", "delta": "ok"},
    ]


def test_parse_flue_sse_events_ignores_heartbeat_comment_lines():
    lines = [
        "event: data",
        'data:[{"type": "message-delta", "kind": "text", "delta": "a"}]',
        "",
        ": heartbeat",
        "",
        "event: data",
        'data:[{"type": "message-delta", "kind": "text", "delta": "b"}]',
        "",
    ]

    assert [c["delta"] for c in parse_flue_sse_events(lines)] == ["a", "b"]


def test_settlement_from_snapshot_matches_only_by_submission_id():
    # A submission's own settlement always carries its own id as submissionId —
    # including one that joined another response, where answeredBySubmissionId
    # additionally points at the host it joined.
    snapshot = {
        "settlements": [
            {"submissionId": "other", "outcome": "completed"},
            {"submissionId": "mine", "outcome": "completed", "answeredBySubmissionId": "host"},
        ]
    }

    assert settlement_from_snapshot(snapshot, "mine") == {
        "submissionId": "mine",
        "outcome": "completed",
        "answeredBySubmissionId": "host",
    }
    assert settlement_from_snapshot(snapshot, "unrelated") is None

    # A later, unrelated submission that joined onto *us* (we are the host) carries
    # our id as answeredBySubmissionId, not submissionId — must not match as ours.
    later_joined_onto_us = {
        "settlements": [{"submissionId": "later", "outcome": "completed", "answeredBySubmissionId": "mine"}]
    }
    assert settlement_from_snapshot(later_joined_onto_us, "mine") is None


def test_serialize_sends_only_the_latest_user_message():
    chat = Chat(size=10)
    chat.add_item(make_user_message("最初の発話"))
    chat.add_item(
        RealtimeConversationItemAssistantMessage(
            type="message",
            role="assistant",
            content=[AssistantContent(type="output_text", text="はい")],
        )
    )
    chat.add_item(make_user_message("次の発話"))
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))

    assert handler._serialize(chat) == "次の発話"


def test_iter_stream_events_maps_conversation_events():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "text", "delta": "こんにちは"},
        {"type": "message-completed"},
        {"type": "submission-settled", "outcome": "completed"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    # Exact-length equality, not just checking the first couple of elements: a
    # regression that flushes the buffer a second time on submission-settled after
    # message-completed already flushed it would otherwise slip through as a
    # trailing empty-content AssistantMessage that this test would never notice.
    assert mapped == [
        TextDelta(text="こんにちは"),
        AssistantMessage(content=[AssistantContent(type="output_text", text="こんにちは")]),
    ]


def test_reasoning_deltas_are_not_spoken():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "reasoning", "delta": "thinking..."},
        {"type": "message-delta", "kind": "text", "delta": "answer"},
        {"type": "message-completed"},
        {"type": "submission-settled", "outcome": "completed"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    # Exact equality: catches reasoning text leaking into the spoken buffer just as
    # much as it catches a raw reasoning TextDelta being yielded.
    assert mapped == [
        TextDelta(text="answer"),
        AssistantMessage(content=[AssistantContent(type="output_text", text="answer")]),
    ]


def test_iter_stream_events_discards_a_stale_messages_buffer_on_an_id_switch():
    # _stream_turn can switch which message id it forwards mid-turn (giving up on
    # a stale leftover, then locking onto its own real message) without ever
    # forwarding a message-completed for the abandoned one — its remaining chunks
    # are simply never forwarded again once the switch happens. Without id
    # tracking here, the two unrelated messages' text would be silently glued
    # into one AssistantMessage instead of the stale half being dropped.
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "text", "messageId": "prev-msg", "delta": "leftover"},
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "real answer"},
        {"type": "message-completed", "messageId": "my-msg"},
        {"type": "submission-settled", "outcome": "completed"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    assert mapped == [
        TextDelta(text="leftover"),
        TextDelta(text="real answer"),
        AssistantMessage(content=[AssistantContent(type="output_text", text="real answer")]),
    ]


def test_message_completed_for_a_different_message_id_does_not_flush_the_buffer():
    # A message-completed that doesn't match the id currently being tracked (the
    # abandoned message's own completion, arriving after this handler already
    # switched to a newer one) must not emit the still-in-progress buffer as a
    # completed AssistantMessage — that message hasn't completed, this
    # unrelated one has. (The buffer is still discarded either way once a
    # message-completed is seen; only whether it gets reported as a completed
    # AssistantMessage on THIS event depends on the id match.)
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "still typing"},
        {"type": "message-completed", "messageId": "other-msg"},
        {"type": "submission-settled", "outcome": "completed"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    assert mapped == [TextDelta(text="still typing")]


def test_settlement_without_a_prior_message_completed_still_flushes_the_buffer():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "text", "delta": "Hello!"},
        {"type": "submission-settled", "outcome": "completed"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    assert mapped == [
        TextDelta(text="Hello!"),
        AssistantMessage(content=[AssistantContent(type="output_text", text="Hello!")]),
    ]


def test_failed_submission_raises():
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [{"type": "submission-settled", "outcome": "failed", "error": {"message": "model unreachable"}}]

    with pytest.raises(RuntimeError, match="model unreachable"):
        list(handler._iter_stream_events(iter(events)))


def test_aborted_submission_ends_the_turn_quietly_without_flushing_the_buffer():
    # Unlike "failed", "aborted" only ever comes from our own abort call (or a
    # barge-in's abort dragging this turn down with it — flue's abort targets the
    # whole conversation instance, not just one submission). Raising here would
    # surface as PROVIDER_FAILURE_FALLBACK for something this adapter itself
    # asked for, so it must end the turn without an exception and without
    # emitting the partial text as a completed AssistantMessage.
    handler = _make_handler(httpx.MockTransport(lambda request: httpx.Response(500)))
    events = [
        {"type": "message-delta", "kind": "text", "delta": "part"},
        {"type": "submission-settled", "outcome": "aborted"},
    ]

    mapped = list(handler._iter_stream_events(iter(events)))

    assert mapped == [TextDelta(text="part")]


def test_stream_turn_does_not_adopt_a_stale_unrelated_submissions_leftover_message():
    # The adapter itself already gave up on a previous turn (deadline/disconnect),
    # but flue kept generating that response in the background, and it is still
    # trickling out when this turn's own read begins. A version that adopts
    # whichever message-bearing chunk it sees first, unconditionally, would lock
    # onto that leftover text and discard this turn's real answer entirely once it
    # arrives (found in an earlier round of review of this same fix). Our own
    # message-started (which always carries our own submissionId for a genuinely
    # fresh response) must override whatever was provisionally locked in, and a
    # submission-settled that turns out to belong to someone else must clear it.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "prev-msg", "submissionId": "prev"}],
                [{"type": "message-delta", "kind": "text", "messageId": "prev-msg", "delta": "leftover"}],
                [{"type": "message-completed", "messageId": "prev-msg"}],
                [{"type": "submission-settled", "submissionId": "prev", "outcome": "completed"}],
                [{"type": "message-started", "messageId": "my-msg", "submissionId": "mine"}],
                [{"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "real answer"}],
                [{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    # The leftover text is unavoidably forwarded — it streams in and is
    # provisionally accepted before anything reveals it isn't ours. What this
    # guards against is losing the real answer once it arrives, not suppressing
    # already-forwarded stray content after the fact.
    assert events == [
        {"type": "message-delta", "kind": "text", "messageId": "prev-msg", "delta": "leftover"},
        {"type": "message-completed", "messageId": "prev-msg"},
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "real answer"},
        {"type": "submission-settled", "submissionId": "mine", "outcome": "completed"},
    ]


def test_stream_turn_overrides_a_stale_message_id_the_instant_our_own_message_started_arrives():
    # Isolates the message-started override specifically: the leftover submission
    # has NOT settled yet (no disqualifying submission-settled/conversation-reset
    # has been seen) when our own fresh message-started arrives, so the
    # reset-on-foreign-settlement fix has no chance to fire first. Without the
    # override, our own real delta would still be silently dropped even with that
    # other fix in place — this was confirmed by temporarily reverting only the
    # override and finding this exact test (and only this one) fails, while
    # test_stream_turn_does_not_adopt_a_stale_unrelated_submissions_leftover_message
    # above kept passing because the leftover happens to settle before our own
    # message-started there — that test alone was not proof this override does
    # anything.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "prev-msg", "submissionId": "prev"}],
                [{"type": "message-delta", "kind": "text", "messageId": "prev-msg", "delta": "leftover"}],
                [{"type": "message-started", "messageId": "my-msg", "submissionId": "mine"}],
                [{"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "real answer"}],
                [{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "real answer"} in events


def test_stream_turn_locks_onto_the_first_message_id_and_ignores_a_later_different_one():
    # Defensive invariant, not a confirmed flue trigger: once a message id has been
    # adopted, chunks bearing any other message id are ignored rather than mixed in.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "my-msg", "submissionId": "mine"}],
                [{"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "mine"}],
                [{"type": "message-delta", "kind": "text", "messageId": "other-msg", "delta": "not mine"}],
                [{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "こんにちは"))

    assert events == [
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "mine"},
        {"type": "submission-settled", "submissionId": "mine", "outcome": "completed"},
    ]


def test_stream_turn_forwards_a_joined_hosts_message():
    # This delivery joined an already-running response instead of starting its own
    # (flue folds a delivery arriving mid-generation into the current response —
    # agent-submissions.ts's join handling): message-started/delta/completed all
    # carry the HOST's submissionId, never "mine". Content must still come
    # through — an earlier version dropped it entirely because message-started was
    # filtered on an exact submissionId match, so message_id never got set and
    # every message-delta/message-completed failed the `message_id is not None`
    # guard. Our own settlement record still carries our own id as submissionId
    # directly (with answeredBySubmissionId pointing at the host), which is what
    # actually ends the turn.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "host-msg", "submissionId": "host"}],
                [{"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "joined answer"}],
                [{"type": "message-completed", "messageId": "host-msg"}],
                [
                    {
                        "type": "submission-settled",
                        "submissionId": "mine",
                        "answeredBySubmissionId": "host",
                        "outcome": "completed",
                    }
                ],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert events == [
        {"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "joined answer"},
        {"type": "message-completed", "messageId": "host-msg"},
        {
            "type": "submission-settled",
            "submissionId": "mine",
            "answeredBySubmissionId": "host",
            "outcome": "completed",
        },
    ]


def test_stream_turn_forwards_a_joined_hosts_message_that_started_before_our_offset():
    # The extreme case of the above: the host message had already started
    # generating before this delivery's admission offset, so even
    # message-started never appears in this read at all — the first thing seen is
    # a mid-flight delta. message_id must still be adopted from it.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "already talking"}],
                [
                    {
                        "type": "submission-settled",
                        "submissionId": "mine",
                        "answeredBySubmissionId": "host",
                        "outcome": "completed",
                    }
                ],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert events[0] == {"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "already talking"}


def test_stream_turn_ignores_a_later_unrelated_submissions_settlement_that_joined_onto_us():
    # We are the host of an in-flight response; a different, later submission
    # joins it before ours finishes. flue settles a joined delivery before the
    # host's own settlement, so that record — carrying OUR id as
    # answeredBySubmissionId, not submissionId — could arrive first. Matching
    # only on submissionId (not also answeredBySubmissionId) means this doesn't
    # mistake it for our own turn ending.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "my-msg", "submissionId": "mine"}],
                [{"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "still going"}],
                [
                    {
                        "type": "submission-settled",
                        "submissionId": "later",
                        "answeredBySubmissionId": "mine",
                        "outcome": "completed",
                    }
                ],
                [{"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": " done"}],
                [{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert events == [
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": "still going"},
        {"type": "message-delta", "kind": "text", "messageId": "my-msg", "delta": " done"},
        {"type": "submission-settled", "submissionId": "mine", "outcome": "completed"},
    ]


def test_stream_turn_recovers_settlement_from_a_conversation_reset():
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [
                    {
                        "type": "conversation-reset",
                        "snapshot": {
                            "settlements": [
                                {"submissionId": "mine", "outcome": "completed"},
                            ]
                        },
                    }
                ],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert events == [{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}]


def test_stream_turn_clears_a_stale_message_id_when_a_conversation_reset_settles_someone_else():
    # Same failure mode as test_stream_turn_does_not_adopt_a_stale_unrelated_submissions_leftover_message,
    # but the leftover submission's settlement arrives folded into a
    # conversation-reset (auto-compaction) instead of its own submission-settled
    # chunk, AND — critically — this turn's own real answer is also a joined
    # response whose message-started never appears (so the *other* fix, adopting
    # our own submissionId's message-started, cannot rescue this case: it never
    # fires). This isolates the conversation-reset branch's own reset logic. A
    # first version of this test used a visible message-started for "our" answer,
    # which passed even with the conversation-reset branch's reset code fully
    # reverted — it was only proving the OTHER fix worked, not this one.
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "prev-msg", "submissionId": "prev"}],
                [{"type": "message-delta", "kind": "text", "messageId": "prev-msg", "delta": "leftover"}],
                [
                    {
                        "type": "conversation-reset",
                        "snapshot": {"settlements": [{"submissionId": "prev", "outcome": "completed"}]},
                    }
                ],
                [{"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "real answer"}],
                [
                    {
                        "type": "submission-settled",
                        "submissionId": "mine",
                        "answeredBySubmissionId": "host",
                        "outcome": "completed",
                    }
                ],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    events = list(handler._stream_turn("session20260919", "hi"))

    assert {"type": "message-delta", "kind": "text", "messageId": "host-msg", "delta": "real answer"} in events
    assert events[-1]["outcome"] == "completed"


def test_stream_turn_raises_when_the_stream_ends_without_settling():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        # The connection closes at a clean frame boundary, but nothing ever settled.
        return httpx.Response(200, content=_sse([{"type": "stream-checkpoint", "incarnation": "inc_1"}]))

    handler = _make_handler(httpx.MockTransport(respond))

    with pytest.raises(RuntimeError, match="ended before the submission settled"):
        list(handler._stream_turn("session20260919", "hi"))

    # Nobody else will ever read this response: flue must be told to stop
    # generating it rather than let it keep running unseen.
    assert "/agents/flue-voice/session20260919/abort" in calls


def test_stream_turn_enforces_a_wall_clock_deadline():
    # Simulate flue's 15s heartbeat well past the deadline: httpx's own read
    # timeout never fires because data keeps arriving, so only the manual
    # deadline check can end this. Bounded (not `while True`): if the deadline
    # check itself were the thing broken, an unbounded generator would make this
    # test hang forever instead of failing.
    def infinite_heartbeats():
        for _ in range(2_000_000):
            yield b": heartbeat\n\n"

    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(200, content=infinite_heartbeats())

    handler = _make_handler(httpx.MockTransport(respond))
    handler.request_timeout_s = 0.05

    with pytest.raises(httpx.ReadTimeout, match="exceeded"):
        list(handler._stream_turn("session20260919", "hi"))

    assert "/agents/flue-voice/session20260919/abort" in calls


def test_stream_turn_does_not_abort_a_submission_that_settled_normally():
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(
            200,
            content=_sse([{"type": "submission-settled", "submissionId": "mine", "outcome": "completed"}]),
        )

    handler = _make_handler(httpx.MockTransport(respond))

    list(handler._stream_turn("session20260919", "hi"))

    assert not any(path.endswith("/abort") for path in calls)


def test_stream_turn_aborts_when_the_caller_closes_the_generator_early():
    # Mirrors what happens when the base class abandons a turn mid-stream (e.g. a
    # newer turn superseding this one during barge-in): the generator is closed
    # without ever being driven to a settlement.
    calls = []

    # Bounded (not `while True`): if a regression stopped the very first chunk
    # from ever being yielded (e.g. the message-delta branch's message_id
    # adoption), an unbounded generator would make `next(gen)` below hang
    # forever instead of failing.
    def infinite_deltas():
        for i in range(10_000):
            yield _sse([{"type": "message-delta", "kind": "text", "messageId": "m", "delta": str(i)}]).encode()

    def respond(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        return httpx.Response(200, content=infinite_deltas())

    handler = _make_handler(httpx.MockTransport(respond))

    gen = handler._stream_turn("session20260919", "hi")
    next(gen)
    gen.close()

    assert "/agents/flue-voice/session20260919/abort" in calls


def test_stream_turn_raises_when_the_admission_post_fails():
    def respond(request):
        if request.method == "POST":
            return httpx.Response(404, json={"error": "unknown agent"})
        raise AssertionError("must not attempt to stream when the admission POST itself failed")

    handler = _make_handler(httpx.MockTransport(respond))

    with pytest.raises(httpx.HTTPStatusError):
        list(handler._stream_turn("session20260919", "hi"))


def test_stream_turn_raises_and_aborts_when_the_stream_get_fails():
    # Symmetric with test_stream_turn_raises_when_the_admission_post_fails above,
    # but for the GET side: flue's stream route can itself return a non-2xx (e.g.
    # a 404 StreamNotFoundError for an offset it no longer has). Unlike the
    # admission POST failing before any submission exists, here one already does
    # — so, unlike that case, this must still tell flue to give up on it.
    calls = []

    def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST" and not request.url.path.endswith("/abort"):
            return httpx.Response(200, json={"offset": "0", "submissionId": "mine", "uid": "inst_1"})
        if request.method == "GET":
            return httpx.Response(404, json={"error": "StreamNotFoundError"})
        return httpx.Response(200)

    handler = _make_handler(httpx.MockTransport(respond))

    with pytest.raises(httpx.HTTPStatusError):
        list(handler._stream_turn("session20260919", "hi"))

    assert any(method == "POST" and path.endswith("/abort") for method, path in calls)


def test_request_posts_to_the_jst_conversation_id_and_streams_its_events(monkeypatch):
    # _request derives the conversation id from the real wall clock. Freezing it
    # (rather than computing the expected id from a second, separate
    # datetime.now() call at the test's own start) avoids a real, if rare, flake:
    # the two reads racing across a JST midnight rollover.
    fixed_now = datetime(2026, 9, 19, 3, 0, 0, tzinfo=timezone.utc)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now

    monkeypatch.setattr(flue_mod, "datetime", _FixedDatetime)
    today_id = conversation_id_for(fixed_now)
    expected_path = f"/agents/flue-voice/{today_id}"
    calls = []

    def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.url.path == expected_path
            assert json.loads(request.content) == {"kind": "user", "body": "こんにちは"}
            return httpx.Response(200, json={"offset": "OFFSET_1", "submissionId": "sub_1", "uid": "inst_1"})
        assert request.url.path == expected_path
        assert request.url.params["offset"] == "OFFSET_1"
        assert request.url.params["live"] == "sse"
        return httpx.Response(
            200,
            content=_sse(
                [{"type": "message-started", "messageId": "m1", "submissionId": "sub_1"}],
                [{"type": "message-delta", "kind": "text", "messageId": "m1", "delta": "はい"}],
                [{"type": "submission-settled", "submissionId": "sub_1", "outcome": "completed"}],
            ),
        )

    handler = _make_handler(httpx.MockTransport(respond))
    handler.agent_name = "flue-voice"

    events = list(handler._request("こんにちは", {}))

    assert {"type": "message-delta", "kind": "text", "messageId": "m1", "delta": "はい"} in events
    assert calls == [("POST", expected_path), ("GET", expected_path)]
