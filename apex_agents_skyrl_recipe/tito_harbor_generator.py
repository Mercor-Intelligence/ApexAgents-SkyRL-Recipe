"""
TITOHarborGenerator — Token-In, Token-Out generator for Harbor tasks.

Unlike the base HarborGenerator which re-tokenizes chat messages, this generator
extracts pre-computed token IDs, loss masks, and logprobs directly from the
archipelago agent's TITO state. This ensures exact token alignment with the
vLLM engine and avoids re-tokenization artifacts.

The agent stores:
  - tito_tokens: List[int]     — full token ID sequence (prompt + all generations + observations)
  - tito_loss_mask: List[int]  — 1 for model-generated tokens, 0 for prompt/observation tokens
  - tito_logprobs: List[float] — per-token logprobs (0.0 for non-generated tokens)
"""

import asyncio
import json
import logging
import os
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import litellm
import ray
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import get_rollout_metrics
from skyrl.train.utils.rate_limiter import create_rate_limiter
from tqdm import tqdm

from apex_agents_skyrl_recipe.metrics_helper import _compute_tool_metrics, finalize_tool_call_metrics

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

MAX_NUM_RETRIES_PER_TRIAL = 2


@ray.remote(num_cpus=0.2, num_gpus=0)
def _run_trial_in_process(config_dict: dict, env_vars: dict):
    """Run a single Harbor trial in an isolated Ray worker process.

    Each invocation gets its own process and event loop, providing full isolation for MCP client
    sessions (anyio streams, httpx connections, SSL contexts).  Unlike threads, separate processes
    avoid C-level segfaults from concurrent SSL/gRPC operations.

    NOTE: This uses 0.2 `num_cpus` . As such the task is I/O bound, but requesting a very tiny `num_cpus`
    for a Ray task is an anti-pattern with Ray.
    """
    # Restore env vars (API keys etc.) forwarded from the entrypoint
    for k, v in env_vars.items():
        os.environ[k] = v

    # Suppress litellm noise — module-level settings from the parent
    # process don't carry over to Ray worker processes.
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    async def _run():
        # Suppress aiohttp "Unclosed client session/connector" errors.
        _suppressed = {"Unclosed client session", "Unclosed connector", "Task was destroyed but it is pending!"}
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(
            lambda loop, ctx: (None if ctx.get("message", "") in _suppressed else loop.default_exception_handler(ctx))
        )

        trial_config = TrialConfig.model_validate(config_dict)
        # Trials now run on any node (not pinned to the entrypoint), and trials_dir is
        # node-local storage — make sure the base dir exists on whatever node runs this
        # trial. (harbor's TrialPaths.mkdir already creates the per-trial subdirs with
        # parents=True; this just guards the base dir / any writes outside TrialPaths.)
        os.makedirs(trial_config.trials_dir, exist_ok=True)
        # OSS harbor deprecated Trial(config) in favor of the async factory.
        trial = await Trial.create(trial_config)
        return await trial.run()

    return asyncio.run(_run())


def _split_lists(time_splits: List[Optional[Dict[str, float]]]) -> Optional[Dict[str, List[float]]]:
    """Per-component lists from per-trajectory time splits, or None if any trajectory lacks them."""
    if not time_splits or any(s is None for s in time_splits):
        return None
    return {name: [s[name] for s in time_splits] for name in time_splits[0]}


def _time_splits_from_metadata(metadata: Optional[dict]) -> Optional[Dict[str, float]]:
    """Aggregate the archipelago agent's per-step timings into an {"llm", "env"} split.

    "llm" is model-inference time and "env" is tool-execution time (the TITO analog of
    ``env.step`` in SkyRLGymGenerator), summed over the trajectory's steps from
    ``metadata["step_timings"]``. Returns None when the agent did not record ``step_timings``,
    so the time-split metrics degrade to absent rather than a misleading zero.
    """
    step_timings = (metadata or {}).get("step_timings")
    if not step_timings:
        return None
    return {
        "llm": sum(s.get("llm_time_sec", 0.0) for s in step_timings),
        "env": sum(s.get("tool_time_sec", 0.0) for s in step_timings),
    }


@dataclass
class TITOHarborAgentOutput:
    prompt_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    reward: float
    stop_reason: str
    trajectory_id: TrajectoryID
    rollout_logprobs: Optional[List[float]] = None
    summarization_count: Optional[int] = None
    num_turns: Optional[int] = None
    # End-to-end wall-clock time (seconds) to generate this trajectory. Optional: may be left
    # as None if timing was not recorded (e.g. failure before timing completes).
    e2e_time: Optional[float] = None
    # Engine/env time split of ``e2e_time``, e.g. {"llm": ..., "env": ...}: "llm" is model-inference
    # time, "env" is tool-execution time. None for trajectories that failed before metadata existed.
    time_splits: Optional[Dict[str, float]] = None
    # Tool-call metrics extracted from the agent's messages. None if the
    # trajectory failed before metadata was available (e.g. agent timeout).
    num_tool_calls: Optional[int] = None
    num_successful_tool_calls: Optional[int] = None
    num_code_exec_tool_calls: Optional[int] = None
    # {server: count} of tool calls per MCP server (keys are _MCP_SERVER_BUCKETS).
    tool_calls_by_server: Optional[dict] = None


class TITOHarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client,
        tokenizer,
        max_seq_len: int,
    ):
        ie_cfg = generator_cfg.inference_engine
        self.generator_cfg = generator_cfg
        self.max_seq_len = max_seq_len
        # Keep the client so we can call end_session() at trajectory teardown,
        # releasing the engine slot used for session-aware load balancing.
        self._inference_engine_client = inference_engine_client

        # Harbor config template
        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # TITO mode: use /v1/completions via litellm's text-completion-openai provider
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"text-completion-openai/{ie_cfg.served_model_name}"
        assert isinstance(inference_engine_client, RemoteInferenceClient)
        base_url = inference_engine_client.proxy_url
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{base_url}/v1"

        # Force TITO mode in harbor config
        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        agent_kwargs["use_tito"] = True

        # Warn if tito_tokenizer_name is missing
        if not agent_kwargs.get("tito_tokenizer_name"):
            logger.warning(
                "tito_tokenizer_name not set in harbor config. "
                "Set harbor_trial_config.agent.kwargs.tito_tokenizer_name=<model_path>"
            )

        logger.info(
            f"TITOHarborGenerator initialized: "
            f"model=text-completion-openai/{ie_cfg.served_model_name}, "
            f"agent={self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"trials_dir={self._harbor_trial_config_template.get('trials_dir', 'trials')}"
        )

        logger.info(f"Full harbor trial config: {self._harbor_trial_config_template}")

        # Rate limiter
        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

        # Eval-time context-budget override (parsed ad-hoc from the generator cfg,
        # same pattern as rate_limit). When set, eval trajectories use this as
        # model_info.max_input_tokens/max_output_tokens instead of the train value.
        train_max_input_tokens = self._harbor_trial_config_template["agent"]["kwargs"]["model_info"]["max_input_tokens"]
        if generator_cfg.eval_max_model_len is not None:
            self._eval_max_model_len = generator_cfg.eval_max_model_len
        else:
            self._eval_max_model_len = train_max_input_tokens

        assert self._eval_max_model_len >= train_max_input_tokens, (
            f"eval_max_model_len ({self._eval_max_model_len}) must be >= than train "
            f"max_input_tokens ({train_max_input_tokens})"
        )
        logger.info(f"Eval max_model_len={self._eval_max_model_len}, train max_model_len={train_max_input_tokens}")

        # Build env vars dict to forward to Ray worker processes
        _forward_keys = {
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "FMP_API_KEY",
            "TERRAPIN_API_KEY",
            "REDUCTO_API_KEY",
            "HF_TOKEN",
            "GOOGLE_API_KEY",
        }
        self._worker_env_vars = {k: v for k, v in os.environ.items() if k in _forward_keys}
        if "GOOGLE_API_KEY" not in self._worker_env_vars:
            logger.warning("GOOGLE_API_KEY is not set; the in-sandbox LLM-judge grading will fail (reward 0).")

        # Pin generation trials to the *inference* nodes (round-robin NodeAffinity), keeping the
        # CPU/IO-bound trial orchestration off the training nodes (whose CPUs are saturated by the
        # Megatron CPU optimizer offload). Round-robin across the inference nodes avoids the old
        # single-node CPU saturation from a fixed soft=False pin. Trial *results* return via Ray
        # object refs (not trials_dir), so placement only affects where orchestration CPU + per-trial
        # log files land — never correctness.
        self._inference_node_ids: List[str] = list(getattr(inference_engine_client, "inference_node_ids", []) or [])
        if not self._inference_node_ids:
            # SkyRL's RemoteInferenceClient (<= 0.3.0) does not expose
            # inference_node_ids (used to pin trial tasks to inference
            # nodes). Fall back to unpinned scheduling:
            # correct everywhere, loses only placement affinity on multi-node
            # clusters (trial tasks may land on training nodes).
            logger.warning(
                "Inference client provides no inference_node_ids; running harbor " "trial tasks without node affinity."
            )
        if self._inference_node_ids:
            logger.info(
                f"Trial tasks pinned (round-robin NodeAffinity) across {len(self._inference_node_ids)} "
                f"inference nodes: {[n[:12] for n in self._inference_node_ids]}"
            )

    def _compute_cache_salt(self) -> Optional[str]:
        """Derive a prefix-cache salt from the current policy version.

        Returns a string keyed on the engine's ``weight_version`` (which advances on each weight sync)
        and the policy model name (so distinct adapters / tenants don't collide). Called once per
        ``generate`` batch so all trajectories share the version at the start of the batch. Returns
        ``None`` when disabled or when the client exposes no weight version. We key on the engine's
        weight version rather than ``global_step`` because in fully-async training they aren't in
        lock-step.
        """
        # Cache salt defaults to true
        weight_version = getattr(self._inference_engine_client, "weight_version")
        assert weight_version is not None, "weight_version is required"

        return f"weight_version={weight_version}"

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        cache_salt = self._compute_cache_salt()
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        # Detect eval vs train from the batch metadata that the SkyRL data path already
        # threads through (prepare_generator_input sets training_phase="eval"/"train").
        batch_metadata = input_batch.get("batch_metadata")
        is_eval = batch_metadata is not None and getattr(batch_metadata, "training_phase", None) == "eval"

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[TITOHarborAgentOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            disable=disable_tqdm,  # disable for fully async training
            total=len(prompts),
            desc="Generating TITO Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id, cache_salt):
            result = await self._run_tito_trial(
                prompt=prompt, trajectory_id=trajectory_id, trial_index=idx, cache_salt=cache_salt, is_eval=is_eval
            )
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id, cache_salt))
        finally:
            progress.close()

        all_outputs, rollout_metrics = self._mask_failed_instances_and_compute_metrics(all_outputs)

        has_logprobs = any(output.rollout_logprobs is not None for output in all_outputs)

        # Per-trajectory end-to-end generation times (one entry per prompt, preserving input order).
        # Omit the field entirely (None) if any trajectory did not record it.
        trajectory_generation_times = [output.e2e_time for output in all_outputs]
        if any(t is None for t in trajectory_generation_times):
            trajectory_generation_times = None

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": [output.response_ids for output in all_outputs],
            "rewards": [output.reward for output in all_outputs],
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": [output.rollout_logprobs for output in all_outputs] if has_logprobs else None,
            "trajectory_generation_times": trajectory_generation_times,
            # Per-trajectory engine/env time splits, aligned 1:1 with responses; None if any
            # trajectory (e.g. a failure) lacks a split.
            "trajectory_time_splits": _split_lists([output.time_splits for output in all_outputs]),
        }

        return generator_output

    async def _run_tito_trial(
        self,
        prompt,
        trajectory_id: TrajectoryID,
        trial_index: int,
        cache_salt: str,
        is_eval: bool = False,
    ) -> TITOHarborAgentOutput:
        """Run a single harbor trial in TITO mode and extract token-level data."""
        reward = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        agent_stop_reason = None
        exc_type = None
        results = None

        trial_start_time = None
        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            # Generate session_id outside the inner try so the `finally`
            # release runs even if config construction raises.
            session_id = uuid4().hex
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
                config["agent"]["kwargs"]["cache_salt"] = cache_salt

                # Eval-time context-budget override: let eval trajectories use a longer
                # context than training. archipelago derives its client-side context cap
                # (_max_context_len) from model_info.max_input_tokens, so overriding it
                # here is sufficient — no harbor-side change needed. The vLLM engine's
                # max_model_len must be >= this value (set in the launch script).
                if is_eval:
                    model_info = config["agent"]["kwargs"]["model_info"]
                    model_info["max_input_tokens"] = self._eval_max_model_len
                    model_info["max_output_tokens"] = self._eval_max_model_len

                # Inject registry_image from archipelago.json if present
                archipelago_json_path = Path(prompt) / "archipelago.json"
                if archipelago_json_path.exists():
                    archipelago_meta = json.loads(archipelago_json_path.read_text())
                    ecr_image = archipelago_meta.get("ecr_image")
                    if ecr_image:
                        config.setdefault("environment", {}).setdefault("kwargs", {})["registry_image"] = ecr_image

                # Build per-trial env vars
                trial_env = dict(self._worker_env_vars)

                async with self._rate_limiter:
                    # Start the trajectory clock at the FIRST entry into the limiter so e2e_time
                    # measures actual run time and EXCLUDES the initial rate-limiter queueing wait.
                    # This makes the group_completion_time vs trajectory_completion_time gap reflect
                    # limiter-induced queueing/serialization rather than the trivial max-of-n order
                    # statistic on run times. Retries that re-acquire the limiter keep the original
                    # start (so inter-retry time is attributed to the trajectory, not dropped).
                    if trial_start_time is None:
                        trial_start_time = time.monotonic()
                    # Round-robin this trial onto one of the inference nodes
                    # (no-op affinity when the client exposes no node ids).
                    task_options = {"enable_task_events": False}
                    if self._inference_node_ids:
                        target_node_id = self._inference_node_ids[trial_index % len(self._inference_node_ids)]
                        task_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                            node_id=target_node_id, soft=True
                        )
                    result_ref = _run_trial_in_process.options(**task_options).remote(config, trial_env)
                    results = await result_ref

                # Parse exception type
                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = (
                    exc_type == "AgentTimeoutError" or exc_type == "Timeout" or exc_type == "TimeoutError"
                )

                if is_agent_timeout_error:
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Exception type: {exc_type}")
                    break
                elif is_context_length_error:
                    logger.debug(f"{prefix} hit ContextLengthExceededError, will train with reward=0")
                    reward = 0
                elif not results.verifier_result:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}")
                    continue
                else:
                    reward = results.verifier_result.rewards["reward"]

                # Validate TITO data exists. All three keys are read outside the
                # retry loop below, so a missing key here must trigger a retry
                # rather than crash the run with a KeyError.
                metadata = results.agent_result.metadata
                missing_tito_keys = [
                    k for k in ("tito_tokens", "tito_loss_mask", "tito_logprobs") if metadata.get(k) is None
                ]
                if missing_tito_keys:
                    logger.warning(f"{prefix} failed: missing TITO keys in metadata: {missing_tito_keys}")
                    continue

                summarization_count = metadata.get("summarization_count", 0)
                num_turns = metadata.get("n_episodes", 0)

                # The agent reports stop_reason="length" when a turn was truncated
                # at its max_tokens cap with no tool call. Treat this as an
                # incomplete trajectory: force reward 0 regardless of the verifier.
                agent_stop_reason = metadata.get("stop_reason")  # noqa: F841 (read after loop)
                if agent_stop_reason == "length":
                    logger.debug(f"{prefix} hit per-turn max_tokens (stop_reason=length), training with reward=0")
                    reward = 0

                successful = True
                logger.debug(f"{prefix} successful: reward={reward}")
                break

            except Exception as e:
                logger.warning(f"{prefix} failed: {e}")
                continue
            finally:
                # Release the session's engine slot for load-balancing, regardless of success /
                # failure / break / continue. New retry attempts get a fresh session_id above.
                # InferenceEngineClient exposes a sync end_session(); the router-backed RemoteInferenceClient
                # exposes async finish_session() instead. Support whichever the client has.
                _end_session = getattr(self._inference_engine_client, "end_session", None)
                if _end_session is not None:
                    _end_session(session_id)
                else:
                    _finish_session = getattr(self._inference_engine_client, "finish_session", None)
                    if _finish_session is not None:
                        await _finish_session(session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            msg = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}, exception type={exc_type}), setting loss mask to [0]."
            if stop_reason == "error":
                msg += f" Results: {results}"
            logger.warning(msg)
            return TITOHarborAgentOutput(
                prompt_ids=[0],
                response_ids=[0],
                loss_mask=[0],
                reward=0,
                stop_reason=stop_reason,
                trajectory_id=trajectory_id,
                rollout_logprobs=[0.0],
                e2e_time=time.monotonic() - trial_start_time if trial_start_time else None,
            )

        # Extract TITO data from metadata
        metadata = results.agent_result.metadata
        tito_tokens = metadata["tito_tokens"]
        tito_loss_mask = metadata["tito_loss_mask"]
        tito_logprobs = metadata["tito_logprobs"]

        # Tool-call metrics (shared success definition with the offline scorer).
        (
            num_tool_calls,
            num_successful_tool_calls,
            num_code_exec_tool_calls,
            tool_calls_by_server,
        ) = _compute_tool_metrics(metadata)

        # Find prompt/response boundary: first loss_mask=1 token starts the response
        try:
            first_gen_idx = next(i for i, m in enumerate(tito_loss_mask) if m == 1)
        except StopIteration:
            logger.warning(f"TITO: no generated tokens in loss_mask for trajectory {trajectory_id}")
            return TITOHarborAgentOutput(
                prompt_ids=[0],
                response_ids=[0],
                loss_mask=[0],
                reward=0,
                stop_reason="error",
                trajectory_id=trajectory_id,
                rollout_logprobs=[0.0],
                e2e_time=time.monotonic() - trial_start_time if trial_start_time else None,
                num_tool_calls=num_tool_calls,
                num_successful_tool_calls=num_successful_tool_calls,
                num_code_exec_tool_calls=num_code_exec_tool_calls,
                tool_calls_by_server=tool_calls_by_server,
            )

        prompt_ids = tito_tokens[:first_gen_idx]
        response_ids = tito_tokens[first_gen_idx:]
        loss_mask = tito_loss_mask[first_gen_idx:]
        logprobs = tito_logprobs[first_gen_idx:]

        # Determine stop reason. "length" (per-turn max_tokens truncation, reward
        # already forced to 0 above) takes priority over context_length so it is
        # not masked out by overlong filtering — we want to train on it.
        max_seq_len = self._eval_max_model_len if is_eval else self.max_seq_len
        max_response_tokens = max(0, max_seq_len - len(prompt_ids))
        if agent_stop_reason == "length":
            stop_reason = "length"
        elif is_context_length_error:
            stop_reason = "context_length"
        elif len(response_ids) > max_response_tokens:
            stop_reason = "context_length"
            logger.warning(
                f"TITO trajectory {trajectory_id}: context length exceeded despite Harbor did not report it."
            )
        else:
            stop_reason = "complete"

        # Apply overlong filtering.
        if self.generator_cfg.apply_overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)

        # Truncate to maximum allowed length. We might have ended with a tool response that
        # made context length exceeded. So truncation is needed.
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        logprobs = logprobs[:max_response_tokens]

        logger.debug(
            f"TITO trajectory {trajectory_id}: prompt={len(prompt_ids)} tokens, "
            f"response={len(response_ids)} tokens, reward={reward}, stop={stop_reason}"
        )

        return TITOHarborAgentOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            reward=reward,
            stop_reason=stop_reason,
            trajectory_id=trajectory_id,
            rollout_logprobs=logprobs,
            summarization_count=summarization_count,
            num_turns=num_turns,
            e2e_time=time.monotonic() - trial_start_time if trial_start_time else None,
            time_splits=_time_splits_from_metadata(metadata),
            num_tool_calls=num_tool_calls,
            num_successful_tool_calls=num_successful_tool_calls,
            num_code_exec_tool_calls=num_code_exec_tool_calls,
            tool_calls_by_server=tool_calls_by_server,
        )

    def _mask_failed_instances_and_compute_metrics(
        self,
        all_outputs: List[TITOHarborAgentOutput],
    ) -> tuple[List[TITOHarborAgentOutput], dict]:
        """Zero out outputs for failed / degenerate instances.

        Policy (per-instance, where "instance" = all n_samples trajectories
        sharing the same instance_id / prompt):

        1. Any trajectory with ``stop_reason == "error"`` (non-timeout
           failure) → mask the entire instance. Stays conservative because
           errors usually indicate the instance itself is broken.

        2. Trajectories with ``stop_reason == "agent_timeout"`` →
           - if count >= ``timeout_mask_instance_threshold`` (default 2):
             mask the entire instance (most of it is unreliable);
           - else: mask only the individual timeout trajectories, keep the
             rest of the instance for training. Counted as
             ``num_instance_with_timeouts_unmasked``.
        """
        timeout_threshold = int(getattr(self.generator_cfg, "timeout_mask_instance_threshold", 2))

        groups: dict = defaultdict(list)
        for output in all_outputs:
            groups[output.trajectory_id.instance_id].append(output)

        num_timeout_trajectories = 0
        num_error_trajectories = 0
        num_fully_masked_instances = 0
        num_instance_with_timeouts_unmasked = 0
        num_traj_masked_individually = 0

        def _zero_out(output: TITOHarborAgentOutput) -> None:
            output.stop_reason = "error"
            output.loss_mask = [0] * len(output.response_ids)
            output.reward = 0
            output.rollout_logprobs = [0.0] * len(output.response_ids)

        for instance_id, group in groups.items():
            # Iterate through each instance. Count the number of timeouts and errors.
            timeouts = [o for o in group if o.stop_reason == "agent_timeout"]
            errors = [o for o in group if o.stop_reason == "error"]
            num_timeout_trajectories += len(timeouts)
            num_error_trajectories += len(errors)

            # Rule 1: any error → mask whole instance (strict).
            if errors:
                for o in group:
                    _zero_out(o)
                num_fully_masked_instances += 1
                continue

            # Rule 2: timeouts.
            if len(timeouts) >= timeout_threshold:
                # Too many timeouts → mask whole instance.
                for o in group:
                    _zero_out(o)
                num_fully_masked_instances += 1
                continue
            elif len(timeouts) > 0:
                # Few timeouts → mask only the timed-out trajectories.
                for o in timeouts:
                    _zero_out(o)
                num_traj_masked_individually += len(timeouts)
                num_instance_with_timeouts_unmasked += 1

        # Compute metrics over still-alive trajectories.
        successful_outputs = [o for o in all_outputs if o.stop_reason != "error"]
        if successful_outputs:
            # Per-trajectory end-to-end generation times; omit entirely if any are unrecorded
            # rather than emit a partially-populated list.
            completion_times = [o.e2e_time for o in successful_outputs]
            if any(t is None for t in completion_times):
                completion_times = None
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
                trajectory_completion_times=completion_times,
                # Per-trajectory engine/env splits over the same successful outputs as
                # completion_times, so the two stay aligned for the overhead-band computation.
                trajectory_time_splits=_split_lists([o.time_splits for o in successful_outputs]),
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if (output.summarization_count or 0) > 0
            )
            rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
                1 for output in successful_outputs if output.stop_reason == "context_length"
            )
            rollout_metrics["generate/trajectories_hit_per_turn_max_tokens"] = sum(
                1 for output in successful_outputs if output.stop_reason == "length"
            )
            rollout_metrics["generate/avg_num_turns"] = sum(
                output.num_turns or 0 for output in successful_outputs
            ) / len(successful_outputs)

            # Tool-call metrics. Aggregate over trajectories that recorded tool
            # data (None for trajectories that failed before metadata existed).
            # We emit RAW additive counts here so ``concatenate_generator_outputs``
            # sums them correctly across generate calls, then derive the
            # micro-averaged rates from those sums via ``finalize_tool_call_metrics``.
            # (The fully-async trainer issues one generate call per trajectory, so
            # emitting the ratios directly and letting the concat heuristic re-fold
            # them would sum per-call ratios and push rates above 1.)
            tool_outputs = [o for o in successful_outputs if o.num_tool_calls is not None]
            if tool_outputs:
                total_tool_calls = sum(o.num_tool_calls for o in tool_outputs)
                total_successful_tool_calls = sum(o.num_successful_tool_calls for o in tool_outputs)
                total_code_exec_tool_calls = sum(o.num_code_exec_tool_calls for o in tool_outputs)
                server_totals = Counter()
                for o in tool_outputs:
                    server_totals.update(o.tool_calls_by_server or {})

                finalize_tool_call_metrics(
                    rollout_metrics,
                    total_tool_calls,
                    total_successful_tool_calls,
                    total_code_exec_tool_calls,
                    len(tool_outputs),
                    server_totals,
                )
        else:
            rollout_metrics = {}

        rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
        rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
        rollout_metrics["generate/num_masked_instances"] = num_fully_masked_instances
        rollout_metrics["generate/num_instance_with_timeouts_unmasked"] = num_instance_with_timeouts_unmasked
        rollout_metrics["generate/num_traj_masked_individually"] = num_traj_masked_individually

        return all_outputs, rollout_metrics
