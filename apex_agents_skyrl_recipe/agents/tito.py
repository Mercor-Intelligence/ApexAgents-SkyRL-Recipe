"""Token-In-Token-Out (TITO) utilities for the archipelago agent.

Provides:
- Fixed-base tokenization for strictly-appending token lists
- Tool call parsing from raw completion text (using vLLM's tool parsers)
- Reasoning extraction (using vLLM's reasoning parsers)
- Conversion from completion responses to chat-like ModelResponse
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litellm.files.main import ModelResponse
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
)

from apex_agents_skyrl_recipe.agents.utils import coerce_stringified_container_args

# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class TITOGenerationResult:
    """Result from a single TITO generation call."""

    model_response: ModelResponse
    output_text: str
    output_token_ids: list[int]
    output_logprobs: list[float]
    output_top_logprobs: list[dict[str, float]]
    # Raw vLLM finish_reason ("stop" | "length" | ...), preserved separately
    # since model_response's is synthesized as "stop"/"tool_calls".
    finish_reason: str | None = None


@dataclass
class TITOTransition:
    """One turn of the TITO agent loop.

    ``input_token_ids`` is the prompt sent to ``/v1/completions`` for this
    step — i.e. the full token list at the time of the call.
    """

    step: int
    input_token_ids: list[int]
    output_token_ids: list[int]
    output_logprobs: list[float]
    output_top_logprobs: list[dict[str, float]]
    output_text: str
    assistant_message: dict[str, Any]
    observation_token_ids: list[int] | None = None


# ── Agent state ──────────────────────────────────────────────────────


class TITOAgentState:
    """Strictly-appending token state for the TITO agent loop.

    Maintains three parallel lists (tokens, loss_mask, logprobs) and a
    list of per-step transitions.  All mutation goes through
    :meth:`record_step`.
    """

    # Constant base for fixed-base observation tokenization.
    # Provides a dummy assistant-with-tool-call so the template renders the
    # subsequent tool message with the correct "previous role" context.
    _FIXED_BASE = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "thinking",
            "tool_calls": [
                {
                    "id": "dummy_0",
                    "type": "function",
                    "function": {"name": "dummy_fn", "arguments": {}},
                }
            ],
        },
    ]

    def __init__(
        self,
        prompt_token_ids: list[int],
        tokenizer,
        tools: list[dict[str, Any]] | None,
    ):
        self.tokens: list[int] = list(prompt_token_ids)
        self.loss_mask: list[int] = [0] * len(prompt_token_ids)
        self.logprobs: list[float] = [0.0] * len(prompt_token_ids)
        self.transitions: list[TITOTransition] = []
        self._tokenizer = tokenizer
        self._tools = tools

        # Load stop token IDs from generation_config.
        # GLM-4.7: [<|endoftext|>(154820), <|user|>(154827), <|observation|>(154829)]
        from transformers import AutoConfig, GenerationConfig

        generation_config_missing = False
        try:
            gc = GenerationConfig.from_pretrained(tokenizer.name_or_path)
        except OSError:
            generation_config_missing = True
            # Some valid model repositories (including Qwen3.5-9B) do not
            # publish a separate generation_config.json. Transformers can
            # still derive generation defaults from config.json. Include the
            # tokenizer EOS as well: Qwen3.5's embedded model default is
            # <|endoftext|>, while its chat template and tokenizer use
            # <|im_end|> as the turn delimiter. Accepting both makes overlap
            # removal correct whichever stop the inference service returns.
            model_config = AutoConfig.from_pretrained(tokenizer.name_or_path)
            gc = GenerationConfig.from_model_config(model_config)
        eos = gc.eos_token_id
        stop_token_ids = set(eos) if isinstance(eos, list) else {eos}
        if generation_config_missing and tokenizer.eos_token_id is not None:
            tokenizer_eos = tokenizer.eos_token_id
            if isinstance(tokenizer_eos, list):
                stop_token_ids.update(tokenizer_eos)
            else:
                stop_token_ids.add(tokenizer_eos)
        stop_token_ids.discard(None)
        if not stop_token_ids:
            raise ValueError(f"No EOS token IDs found for tokenizer {tokenizer.name_or_path!r}")
        self._stop_token_ids: set[int] = stop_token_ids

        # Pre-compute fixed base token IDs (constant across all steps)
        template_kwargs: dict[str, Any] = {}
        if tools:
            template_kwargs["tools"] = tools
        self._fixed_base_ids = _apply_chat_template_to_ids(
            tokenizer,
            self._FIXED_BASE,
            add_generation_prompt=False,
            **template_kwargs,
        )

    def record_step(
        self,
        result: TITOGenerationResult,
        step: int,
        assistant_message: dict[str, Any],
        tool_response_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record one agent step: generation + optional observation.

        Appends generation tokens (mask=1), then computes observation
        tokens using the fixed-base approach (O(1) per step, no
        re-tokenization of prior conversation).

        The model generates a stop token (e.g. ``<|observation|>``) as its
        last token. We keep it in generation (mask=1, real logprob). The
        fixed-base delta also starts with the same token, so we strip it.

        Args:
            result: Output from generate_response_tito().
            step: Step index.
            assistant_message: The assistant message dict built from the response.
            tool_response_messages: Raw tool response messages from MCP execution.
                None when the model did not call any tools (final step).
        """
        from apex_agents_skyrl_recipe.agents.utils import _normalize_tool_messages_for_tito

        # Snapshot input_token_ids BEFORE mutating self.tokens
        input_token_ids = list(self.tokens)

        # Append generation (mask=1, logprobs=actual)
        n_gen = len(result.output_token_ids)
        self.tokens.extend(result.output_token_ids)
        self.loss_mask.extend([1] * n_gen)
        self.logprobs.extend(result.output_logprobs)

        # Tokenize observation via fixed-base approach (O(1) per step).
        # Uses a constant dummy base with a dummy assistant+tool_call to
        # provide the correct "previous role" context for the template.
        # The delta = tokenize(base + tool_msgs, gen_prompt=True) - base_ids.
        obs_token_ids = None
        if tool_response_messages is not None:
            normalized_tool_msgs = _normalize_tool_messages_for_tito(tool_response_messages)

            template_kwargs: dict[str, Any] = {}
            if self._tools:
                template_kwargs["tools"] = self._tools

            full_ids = _apply_chat_template_to_ids(
                self._tokenizer,
                self._FIXED_BASE + normalized_tool_msgs,
                add_generation_prompt=True,
                **template_kwargs,
            )

            # Compute the observation delta, handling stop-token overlap. The
            # model emits a stop token (e.g. <|observation|> or <|im_end|>) as
            # its last generated token (kept in generation, mask=1). The
            # observation is the canonical tokens that follow that stop token in
            # a fresh render. Where the stop token sits relative to the fixed
            # base differs by model, giving two cases:
            stop = result.output_token_ids[-1] if result.output_token_ids else None
            base_ids = self._fixed_base_ids
            cut = len(base_ids)

            # Case B — trailing overlap (ChatML / Qwen3.x):
            # The stop token <|im_end|> CLOSES the assistant turn, so it lives
            # inside base_ids, followed by template-only suffix tokens (e.g. the
            # "\n" after <|im_end|>) that actually belong to the observation.
            # The model stops exactly AT <|im_end|> and never emits that "\n",
            # so the naive delta full_ids[len(base):] would silently drop it
            # (TITO stream "<|im_end|><|im_start|>" vs canonical
            # "<|im_end|>\n<|im_start|>"). Re-anchor the delta at the assistant's
            # closing stop token so the suffix is recovered; the Case-A strip
            # below then removes the duplicated stop token.
            if (
                stop is not None
                and stop in self._stop_token_ids
                and base_ids
                and base_ids[-1] != stop
                and stop in base_ids
            ):
                cut = len(base_ids) - 1 - base_ids[::-1].index(stop)

            obs_token_ids = full_ids[cut:]

            # Case A — leading overlap (GLM-4.7, and Qwen after re-anchoring):
            # The delta begins with the stop token the model already generated.
            # For GLM-4.7 the template STARTS the tool section with the stop
            # token <|observation|>; for Qwen the re-anchor above made the delta
            # begin with <|im_end|>. Strip it so it isn't doubled. We check the
            # actual overlap (delta[0] == model's last token) rather than mere
            # set membership, to avoid stripping when the model used a different
            # stop token or finished without one.
            if (
                obs_token_ids
                and result.output_token_ids
                and result.output_token_ids[-1] == obs_token_ids[0]
                and obs_token_ids[0] in self._stop_token_ids
            ):
                obs_token_ids = obs_token_ids[1:]

            n_obs = len(obs_token_ids)
            self.tokens.extend(obs_token_ids)
            self.loss_mask.extend([0] * n_obs)
            self.logprobs.extend([0.0] * n_obs)

        self.transitions.append(
            TITOTransition(
                step=step,
                input_token_ids=input_token_ids,
                output_token_ids=result.output_token_ids,
                output_logprobs=result.output_logprobs,
                output_top_logprobs=result.output_top_logprobs,
                output_text=result.output_text,
                assistant_message=assistant_message,
                observation_token_ids=obs_token_ids,
            )
        )

    def check_invariants(self) -> None:
        """Assert the three lists have equal length."""
        assert len(self.tokens) == len(self.loss_mask) == len(self.logprobs), (
            f"TITO parallel list length mismatch: "
            f"tokens={len(self.tokens)}, mask={len(self.loss_mask)}, logprobs={len(self.logprobs)}"
        )

    def save_transitions(self, path: Path) -> None:
        """Dump transitions to a JSON file."""
        from dataclasses import asdict

        data = []
        for t in self.transitions:
            entry = asdict(t)
            # Add convenience length fields
            entry["input_len"] = len(t.input_token_ids)
            entry["output_len"] = len(t.output_token_ids)
            if t.observation_token_ids is not None:
                entry["observation_len"] = len(t.observation_token_ids)
            data.append(entry)
        path.write_text(json.dumps(data, indent=2, default=str))

    def save_debug(self, path: Path) -> None:
        """Dump the complete token stream decoded to a string for human inspection.

        This is the full ``self.tokens`` (prompt + all generation/observation
        rounds) decoded with special tokens visible, so issues like double
        ``<|observation|>`` are immediately obvious.
        """
        text = self._tokenizer.decode(self.tokens, skip_special_tokens=False)
        path.write_text(text)


# ── Tokenization helpers ─────────────────────────────────────────


def tokenize_initial_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tokenizer,
    chat_template: str | None = None,
) -> list[int]:
    """Tokenize the initial messages (system + user) with tools, plus generation prompt.

    This is the starting point of the token list — everything before the first
    assistant generation.
    """
    token_ids = _apply_chat_template_to_ids(
        tokenizer,
        messages,
        tools=tools if tools else None,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    return token_ids


def _apply_chat_template_to_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    **kwargs,
) -> list[int]:
    """Apply chat template and return a flat list of token IDs.

    Renders to text first then encodes, which is reliable across
    transformers versions (v4.x returns list[int], v5.x returns BatchEncoding).
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        **kwargs,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids


def _build_chat_completion_request(tools: list[dict[str, Any]] | None = None):
    """Build a minimal ChatCompletionRequest for vLLM parsers."""
    from vllm.entrypoints.openai.chat_completion.protocol import (  # type: ignore[import-untyped]
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )

    tools_param = None
    if tools:
        tools_param = [ChatCompletionToolsParam.model_validate(t) for t in tools]

    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "dummy"}],
        tools=tools_param,
    )


# ── Tool call parsing ────────────────────────────────────────────────


@dataclass
class ToolParseResult:
    """Result of parsing tool calls from raw completion text."""

    content: str
    tool_calls: list[dict[str, Any]]  # Each has 'name' and 'arguments' (JSON string)
    has_tool_calls: bool
    parse_error: bool = False


def parse_tool_calls_from_text(
    text: str,
    tools: list[dict[str, Any]] | None = None,
    parser_name: str = "glm47",
    tokenizer=None,
) -> ToolParseResult:
    """Parse tool calls from raw completion text using vLLM's tool parser.

    Args:
        text: Raw text output from /v1/completions.
        tools: Tool definitions for type-aware argument deserialization.
        parser_name: vLLM tool parser name (default: "glm47").
        tokenizer: HuggingFace tokenizer (required by vLLM parsers).

    Returns:
        ToolParseResult with parsed content and tool calls.
    """
    # Check for truncated tool calls (started but not closed)
    has_open_tag = "<tool_call>" in text
    has_close_tag = "</tool_call>" in text
    if has_open_tag and not has_close_tag:
        content = text[: text.find("<tool_call>")]
        return ToolParseResult(
            content=content,
            tool_calls=[],
            has_tool_calls=False,
            parse_error=True,
        )

    if not has_open_tag:
        return ToolParseResult(
            content=text,
            tool_calls=[],
            has_tool_calls=False,
        )

    # Use vLLM's parser
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager  # type: ignore[import-untyped]

    parser = ToolParserManager.get_tool_parser(parser_name)(tokenizer)
    request = _build_chat_completion_request(tools)
    extracted = parser.extract_tool_calls(text, request)

    if extracted.tools_called and extracted.tool_calls:
        parsed_calls = []
        for tc in extracted.tool_calls:
            # vLLM's XML tool parsers usually type args from the (flattened) tool
            # schema, but re-hydrate any top-level container that still arrived
            # JSON-stringified, as a belt-and-suspenders before tool validation.
            # Envelope wrapping is handled at dispatch via flatten_envelope_tools
            # + rewrap_envelope_arguments, not here.
            arguments = coerce_stringified_container_args(tc.function.name, tc.function.arguments, tools)
            parsed_calls.append(
                {
                    "name": tc.function.name,
                    "arguments": arguments,  # JSON string (containers re-hydrated)
                }
            )
        return ToolParseResult(
            content=extracted.content or "",
            tool_calls=parsed_calls,
            has_tool_calls=True,
        )

    return ToolParseResult(
        content=extracted.content or text,
        tool_calls=[],
        has_tool_calls=False,
    )


# ── Reasoning extraction ─────────────────────────────────────────────


def extract_reasoning(
    text: str,
    reasoning_parser_name: str = "glm45",
    tokenizer=None,
) -> tuple[str | None, str | None]:
    """Extract reasoning and visible content from raw model output.

    Uses vLLM's reasoning parser to split ``<think>...</think>`` blocks.

    Args:
        text: Raw text output from /v1/completions.
        reasoning_parser_name: vLLM reasoning parser name (default: "glm45").
        tokenizer: HuggingFace tokenizer (required by vLLM parsers).

    Returns:
        (reasoning_content, visible_content) — either may be None.
    """
    from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager  # type: ignore[import-untyped]

    parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(tokenizer)
    request = _build_chat_completion_request()
    reasoning_content, visible_content = parser.extract_reasoning(text, request)
    return reasoning_content, visible_content


# ── Completion → ModelResponse conversion ────────────────────────────


def completion_to_model_response(
    completion_response,
    tools: list[dict[str, Any]] | None = None,
    tool_call_parser: str = "glm47",
    reasoning_parser: str = "glm45",
    tokenizer=None,
) -> ModelResponse:
    """Convert a text completion response to a chat-like ModelResponse.

    Uses vLLM's tool parser and reasoning parser to extract tool calls and
    thinking blocks from the raw text, then constructs a ModelResponse that
    looks like it came from litellm.acompletion() with tool calling support.

    Args:
        completion_response: Response from litellm.atext_completion().
        tools: Tool definitions for type-aware argument parsing.
        tool_call_parser: vLLM tool parser name (default: "glm47").
        reasoning_parser: vLLM reasoning parser name (default: "glm45").
        tokenizer: HuggingFace tokenizer for the parsers.

    Returns:
        ModelResponse mimicking a /chat/completions response.
    """
    choice = completion_response.choices[0]
    raw_text = choice.text

    # Extract reasoning via vLLM's reasoning parser
    reasoning_content, visible_text = extract_reasoning(
        raw_text,
        reasoning_parser_name=reasoning_parser,
        tokenizer=tokenizer,
    )

    # Parse tool calls from the visible text (after reasoning extraction)
    # The tool parser should see the text WITHOUT the <think>...</think> wrapper
    text_for_tool_parsing = visible_text if visible_text is not None else raw_text
    parse_result = parse_tool_calls_from_text(
        text_for_tool_parsing,
        tools,
        parser_name=tool_call_parser,
        tokenizer=tokenizer,
    )

    # Build the message
    content = parse_result.content.strip() if parse_result.content.strip() else None
    message_kwargs: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if reasoning_content:
        message_kwargs["reasoning_content"] = reasoning_content

    # Add tool calls if present
    finish_reason = "stop"
    if parse_result.has_tool_calls:
        finish_reason = "tool_calls"
        tool_calls_list = []
        for i, tc in enumerate(parse_result.tool_calls):
            tool_calls_list.append(
                ChatCompletionMessageToolCall(
                    id=f"call_{i}",
                    type="function",
                    function=Function(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
            )
        message_kwargs["tool_calls"] = tool_calls_list

    message = Message(**message_kwargs)

    usage = getattr(completion_response, "usage", None)
    model_response = ModelResponse(
        id=getattr(completion_response, "id", ""),
        choices=[
            Choices(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        model=getattr(completion_response, "model", ""),
        usage=usage,
    )

    return model_response
