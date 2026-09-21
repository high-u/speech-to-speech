from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class FlueLanguageModelHandlerArguments(LanguageModelBaseArguments):
    """Arguments for the ``flue`` LLM backend.

    The agent server owns the conversation history, the model selection and the
    tool execution, so the shared history options (``--chat_size``,
    ``--init_chat_prompt``) and ``--model_name`` are not sent to it.

    flue has no session-listing API, so there is nothing to ask "what was the most
    recent session for this agent". The conversation id is a random uuid4, minted
    the first time this handler is used and again whenever at least
    ``--flue_session_gap_hours`` has passed since the previous turn.
    """

    flue_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL where the flue app's agent router is mounted, e.g. 'http://127.0.0.1:5173'. Required."
        },
    )
    flue_agent_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Mount path segment of the flue agent that answers (the <name> in "
            "'/agents/<name>/<conversationId>'). This is not resolved or validated at "
            "startup: flue has no agent catalog endpoint, so an unknown name only fails "
            "on the first turn. Required."
        },
    )
    flue_request_timeout_s: float = field(
        default=300.0,
        metadata={
            "help": "Wall-clock deadline in seconds for one turn, from the first request to the "
            "submission settling. flue's event stream sends a heartbeat every 15s even while a "
            "tool runs, so this is enforced as an explicit deadline rather than an HTTP read "
            "timeout (which would never fire on its own). Default is 300.0."
        },
    )
    compact_history: bool = field(
        default=False,
        metadata={"help": "Not supported by the flue backend: the agent server owns the history."},
    )
    flue_session_gap_hours: float = field(
        default=0.0,
        metadata={
            "help": "Hours of silence since the last turn after which a new conversation id is minted. "
            "0 (default) starts a new conversation on every turn."
        },
    )
