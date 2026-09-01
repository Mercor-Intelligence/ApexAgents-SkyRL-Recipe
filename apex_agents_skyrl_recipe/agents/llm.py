"""LLM call utilities for the archipelago agent.

Wraps litellm.acompletion with retry logic for timeouts, rate limits,
and transient errors. Retries are handled here so the agent loop stays clean.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import litellm
from litellm.files.main import ModelResponse

if TYPE_CHECKING:
    from apex_agents_skyrl_recipe.agents.tito import TITOGenerationResult
from harbor.llms.base import ContextLengthExceededError
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

# ── Exception helpers ────────────────────────────────────────────────


def _is_context_window_error(e: Exception) -> bool:
    """Detect context window exceeded errors regardless of exception type.

    LiteLLM sometimes raises ``ContextWindowExceededError`` directly, but
    many providers (Gemini, vLLM, etc.) return context-length errors as
    ``BadRequestError`` instead.  This catches both by checking the
    exception type *and* the error message string.
    """
    if isinstance(e, ContextWindowExceededError):
        return True
    error_str = str(e).lower()
    patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
        "model's context length",
    ]
    return any(p in error_str for p in patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """Detect deterministic BadRequestErrors that will always fail."""
    error_str = str(e).lower()
    patterns = [
        "tools are supported",
        "too many tools",
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
    ]
    return any(p in error_str for p in patterns)


def _should_not_retry(e: Exception) -> bool:
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


def _handle_litellm_error(e: Exception) -> None:
    """Translate litellm exceptions into harbor exceptions then re-raise.

    Converts context-window errors into Harbor's ``ContextLengthExceededError``
    so that trial.py can record ``ExceptionInfo`` for downstream use
    (e.g. overlong filtering in RL).  Always re-raises; never returns normally.
    """
    if _is_context_window_error(e):
        raise ContextLengthExceededError from e
    raise e


def _is_rate_limit(e: Exception) -> bool:
    """Detect rate limit errors including provider-specific patterns."""
    if isinstance(e, RateLimitError):
        return True
    return "429" in str(e) or "RateLimitError" in type(e).__name__ or "RESOURCE_EXHAUSTED" in str(e)


# ── Retry decorator ─────────────────────────────────────────────────


def with_llm_retry(
    logger: logging.Logger,
    max_timeout_retries: int,
    max_rate_limit_retries: int,
    max_transient_retries: int,
    timeout_backoff: float = 5.0,
    rate_limit_backoff: float = 30.0,
    transient_backoff: float = 5.0,
    jitter: float = 5.0,
):
    """Decorator that retries LLM calls with separate limits for different failure modes.

    Three categories of retriable errors, each with independent counters:
      - **Timeout** (``litellm.Timeout``): server didn't respond in time.
      - **Rate limit** (429 / ``RateLimitError`` / ``RESOURCE_EXHAUSTED``):
        concurrency or quota exceeded.
      - **Transient** (``ServiceUnavailableError``, ``APIConnectionError``,
        ``InternalServerError``, ``BadGatewayError``): temporary server issues.

    Non-retriable errors (context window exceeded, auth failures, etc.)
    are raised immediately.

    Counters are per-call (reset on each invocation of the decorated function),
    not per-trial. They accumulate across error types — e.g. alternating between
    timeouts and rate limits still counts toward each limit independently.

    Args:
        logger: The agent's trial logger. Retry messages go to trial.log.
        max_timeout_retries: Max timeout retries per call before giving up.
        max_rate_limit_retries: Max rate-limit retries per call before giving up.
        max_transient_retries: Max transient-error retries per call before giving up.
        timeout_backoff: Base backoff (seconds) after a timeout.
        rate_limit_backoff: Base backoff (seconds) after a rate limit.
        transient_backoff: Base backoff (seconds) after a transient error.
        jitter: Max random jitter (seconds) added to each backoff.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            num_timeouts = 0
            num_rate_limits = 0
            num_transient = 0

            while True:
                try:
                    result = await func(*args, **kwargs)
                    return result

                except Exception as e:
                    # ── Non-retriable: raise immediately ──────────────
                    if isinstance(e, ContextWindowExceededError) or _should_not_retry(e):
                        _handle_litellm_error(e)  # always re-raises

                    # ── Timeout ───────────────────────────────────────
                    if isinstance(e, Timeout):
                        num_timeouts += 1
                        if num_timeouts > max_timeout_retries:
                            logger.error(
                                f"LLM timed out {num_timeouts} times " f"(max {max_timeout_retries}), giving up"
                            )
                            _handle_litellm_error(e)
                        delay = timeout_backoff + random.uniform(0, jitter)
                        logger.warning(
                            f"LLM timeout ({num_timeouts}/{max_timeout_retries}), " f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # ── Rate limit ────────────────────────────────────
                    if _is_rate_limit(e):
                        num_rate_limits += 1
                        if num_rate_limits > max_rate_limit_retries:
                            logger.error(
                                f"Rate limited {num_rate_limits} times " f"(max {max_rate_limit_retries}), giving up"
                            )
                            _handle_litellm_error(e)
                        delay = rate_limit_backoff + random.uniform(0, jitter)
                        logger.warning(
                            f"Rate limited ({num_rate_limits}/{max_rate_limit_retries}), " f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # ── Transient server errors ───────────────────────
                    if isinstance(
                        e,
                        (
                            ServiceUnavailableError,
                            APIConnectionError,
                            InternalServerError,
                            BadGatewayError,
                        ),
                    ):
                        num_transient += 1
                        if num_transient > max_transient_retries:
                            logger.error(
                                f"Transient error {num_transient} times "
                                f"(max {max_transient_retries}), giving up: {repr(e)}"
                            )
                            _handle_litellm_error(e)
                        delay = transient_backoff * (2 ** (num_transient - 1)) + random.uniform(0, jitter)
                        logger.warning(
                            f"Transient error ({num_transient}/{max_transient_retries}), "
                            f"retrying in {delay:.1f}s: {repr(e)}"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # ── Unknown / non-retriable ───────────────────────
                    _handle_litellm_error(e)

        return wrapper

    return decorator


# ── Public API ───────────────────────────────────────────────────────


async def generate_response(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    collect_rollout_details: bool = False,
    llm_kwargs: dict[str, Any],
) -> ModelResponse:
    """Call ``litellm.acompletion`` with sane defaults.

    All retry logic is handled by the ``@with_llm_retry`` decorator —
    callers get back a ``ModelResponse`` or a raised non-retriable exception.

    Args:
        model: LiteLLM model identifier.
        messages: Conversation history.
        tools: OpenAI-format tool definitions (or None).
        temperature: Sampling temperature.
        api_base: Optional API base URL override.
        collect_rollout_details: If True, request logprobs + token IDs.
        llm_kwargs: Arbitrary extra kwargs forwarded to ``acompletion``
            (e.g. ``max_retries``, ``timeout``, ``reasoning_effort``).

    Returns:
        ``litellm.ModelResponse``.

    Raises:
        `harbor.llms.base.ContextLengthExceededError`: If the context window is exceeded. This is
            rised to archipelago level, then to Trial.run() level so the Trial results can
            record the error. This info is needed by RL (e.g. overlong filtering).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools if tools else None,
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    kwargs.update(llm_kwargs)

    # Default to 0 internal retries — we handle retries ourselves.
    # In LiteLLM, num_retries silently overrides max_retries if both are
    # present, so we must zero out both to prevent accidental double-retry.
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0

    return await litellm.acompletion(**kwargs)


# ── TITO sampling backends ────────────────────────────────────────────
# Both backends are "token-in, token-out": send the running token sequence,
# get back sampled token IDs + per-token logprobs. The only difference is the
# transport. Downstream parsing (tool calls / reasoning) runs on the decoded
# text and is identical for both.
#
#   openai_completions : litellm.atext_completion -> {api_base}/completions.
#       Works for local vLLM AND Fireworks. Both return output token IDs
#       (vLLM via return_token_ids -> choice.token_ids; Fireworks via
#       logprobs.token_ids, which litellm also surfaces as choice.token_ids)
#       plus token_logprobs + top_logprobs.
#   tinker : tinker SDK SamplingClient.sample_async(ModelInput.from_ints(...)).
#       Returns sequences[0].tokens + .logprobs (no top_logprobs — we synthesize
#       a top-1 dict so downstream length invariants hold).


@dataclass
class _TITOSample:
    """Backend-normalized sample: what both backends must produce."""

    raw_text: str
    token_ids: list[int]
    logprobs: list[float]
    top_logprobs: list[dict]
    finish_reason: str | None
    usage: Any = None


def _tito_max_tokens(messages_tokens: list[int], max_context_len: int, llm_kwargs: dict) -> int:
    """max_tokens = remaining context budget, optionally capped by a caller value."""
    context_budget = max(1, max_context_len - len(messages_tokens))
    requested = llm_kwargs.get("max_tokens")
    return min(requested, context_budget) if requested is not None else context_budget


async def _sample_openai_completions(
    *,
    model,
    messages_tokens,
    max_tokens,
    api_base,
    api_key,
    llm_kwargs,
) -> _TITOSample:
    kwargs: dict[str, Any] = {"model": model, "prompt": messages_tokens}
    if api_base is not None:
        kwargs["api_base"] = api_base
        # Local vLLM ignores the key; Fireworks/hosted needs the real one.
        kwargs["api_key"] = api_key or "dummy"
    kwargs.update(llm_kwargs)
    # top_k=-1 means "disabled / full vocab" in vLLM (and SkyRL configs), but
    # Fireworks requires top_k in [0, 100] and uses 0 for disabled. Translate so a
    # SkyRL-style top_k=-1 samples over the full vocab on Fireworks too (rather than
    # falling back to Fireworks' finite default of 40). vLLM ignores this (accepts -1).
    if api_base and "fireworks" in api_base and kwargs.get("top_k") == -1:
        kwargs["top_k"] = 0
    kwargs["return_token_ids"] = True  # vLLM: fill choice.token_ids
    kwargs["max_retries"] = 0
    kwargs["num_retries"] = 0
    kwargs.setdefault("logprobs", 5)  # Fireworks: fills logprobs.token_ids
    kwargs["max_tokens"] = max_tokens  # set last so it wins over llm_kwargs

    response = await litellm.atext_completion(**kwargs)
    choice = response.choices[0]
    assert getattr(choice, "token_ids", None), f"No token_ids in response: {choice}"
    token_ids = list(choice.token_ids)

    lp = choice.logprobs
    assert lp and getattr(lp, "token_logprobs", None), f"No token_logprobs: {lp}"
    assert all(p is not None for p in lp.token_logprobs), f"None token_logprobs: {lp}"
    logprobs = [float(p) for p in lp.token_logprobs]
    assert getattr(lp, "top_logprobs", None) and all(d is not None for d in lp.top_logprobs), f"No top_logprobs: {lp}"
    top_logprobs = [dict(d) for d in lp.top_logprobs]
    assert (
        len(token_ids) == len(logprobs) == len(top_logprobs)
    ), f"Length mismatch: {len(token_ids)} != {len(logprobs)} != {len(top_logprobs)}"
    return _TITOSample(
        raw_text=choice.text,
        token_ids=token_ids,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=getattr(response, "usage", None),
    )


async def _sample_tinker(
    *,
    tinker_sampling_client,
    messages_tokens,
    max_tokens,
    llm_kwargs,
    tokenizer,
) -> _TITOSample:
    if tinker_sampling_client is None:
        raise RuntimeError("tito_backend='tinker' but no tinker_sampling_client was provided.")
    if tokenizer is None:
        raise RuntimeError("tito_backend='tinker' requires a tokenizer to decode sampled tokens.")
    from tinker.types import SamplingParams
    from tinker.types.model_input import ModelInput

    # Tinker's SamplingParams supports top_k (default -1 = full vocab / disabled,
    # matching vLLM/SkyRL semantics); it has no min_p. Forward top_k so a config
    # top_k=-1 samples over the full vocab (the "top_k 0" full-vocab setting).
    sp = SamplingParams(
        temperature=llm_kwargs.get("temperature", 0.0),
        top_p=llm_kwargs.get("top_p", 1.0),
        top_k=llm_kwargs.get("top_k", -1),
        max_tokens=max_tokens,
    )
    output = await tinker_sampling_client.sample_async(
        prompt=ModelInput.from_ints(tokens=list(messages_tokens)),
        num_samples=1,
        sampling_params=sp,
    )
    seq = output.sequences[0]
    token_ids = list(seq.tokens)
    raw_lp = getattr(seq, "logprobs", None)
    logprobs = [float(x) for x in raw_lp] if raw_lp is not None else [0.0] * len(token_ids)
    # Tinker returns no top_logprobs; synthesize a top-1 map so length invariants hold.
    top_logprobs = [{str(t): lp} for t, lp in zip(token_ids, logprobs)]
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    sr = getattr(seq, "stop_reason", None)
    finish_reason = "length" if sr and "length" in str(sr).lower() else "stop"
    assert (
        len(token_ids) == len(logprobs) == len(top_logprobs)
    ), f"Length mismatch: {len(token_ids)} != {len(logprobs)} != {len(top_logprobs)}"
    return _TITOSample(
        raw_text=raw_text,
        token_ids=token_ids,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        finish_reason=finish_reason,
        usage=None,
    )


async def generate_response_tito(
    *,
    model: str,
    messages_tokens: list[int],
    max_context_len: int,
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    llm_kwargs: dict[str, Any],
    tokenizer=None,
    tool_call_parser: str = "glm47",
    reasoning_parser: str = "glm45",
    backend: str = "openai_completions",
    tinker_sampling_client=None,
    logger: logging.Logger | None = None,
) -> "TITOGenerationResult":
    """Call vLLM's ``/v1/completions`` endpoint with token IDs (TITO mode).

    Uses ``litellm.atext_completion`` to send raw token IDs and get back
    raw token IDs + logprobs, then uses vLLM's tool and reasoning parsers
    to extract tool calls and thinking blocks from the decoded text.

    Args:
        model: LiteLLM model identifier (e.g. ``text-completion-openai/model-name``).
        messages_tokens: The full strictly-appending token list so far.
        max_context_len: Model's maximum context length. Used to compute
            ``max_tokens = max_context_len - len(messages_tokens)``.
            Unlike ``/chat/completions``, ``/v1/completions`` defaults to
            ``max_tokens=16`` if not set, so we must compute it dynamically.
            Pass ``max_tokens`` in ``llm_kwargs`` to cap each turn instead; a
            turn that hits that cap comes back with ``finish_reason == "length"``.
        tools: OpenAI-format tool definitions (for parsing tool calls from output).
        api_base: API base URL for the vLLM server.
        llm_kwargs: Extra kwargs (temperature, timeout, etc.).
        tokenizer: HuggingFace tokenizer (required by vLLM tool parsers).
        tool_call_parser: vLLM tool parser name (default: "glm47").

    Returns:
        TITOGenerationResult with model_response, output token IDs,
        logprobs, and top_logprobs.
    """
    from apex_agents_skyrl_recipe.agents.tito import (
        TITOGenerationResult,
        completion_to_model_response,
    )

    # /v1/completions (and Tinker) need max_tokens set explicitly; always cap to
    # the remaining context budget so prompt + max_tokens never exceeds the window.
    max_tokens = _tito_max_tokens(messages_tokens, max_context_len, llm_kwargs)

    # --- backend dispatch: produce a backend-normalized _TITOSample -----------
    if backend == "tinker":
        sample = await _sample_tinker(
            tinker_sampling_client=tinker_sampling_client,
            messages_tokens=messages_tokens,
            max_tokens=max_tokens,
            llm_kwargs=llm_kwargs,
            tokenizer=tokenizer,
        )
    elif backend == "openai_completions":
        sample = await _sample_openai_completions(
            model=model,
            messages_tokens=messages_tokens,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            llm_kwargs=llm_kwargs,
        )
    else:
        raise ValueError(f"Unknown tito_backend {backend!r}; expected 'openai_completions' or 'tinker'.")

    # --- shared post-processing: parse tool calls + reasoning from the text ---
    # completion_to_model_response only reads choices[0].text and .usage, so a
    # light shim works for both backends (Tinker has no litellm response object).
    shim = SimpleNamespace(
        choices=[SimpleNamespace(text=sample.raw_text)],
        usage=sample.usage,
    )

    # Parse tool calls + reasoning via vLLM's parsers, convert to ModelResponse
    model_response = completion_to_model_response(
        shim,
        tools,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        tokenizer=tokenizer,
    )

    return TITOGenerationResult(
        model_response=model_response,
        output_text=sample.raw_text,
        output_token_ids=sample.token_ids,
        output_logprobs=sample.logprobs,
        output_top_logprobs=sample.top_logprobs,
        finish_reason=sample.finish_reason,
    )
