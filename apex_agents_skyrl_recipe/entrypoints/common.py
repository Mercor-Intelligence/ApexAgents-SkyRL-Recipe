"""Shared config, utilities, and base experiment class for TITO Harbor entrypoints."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import yaml
from loguru import logger
from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.rate_limiter import RateLimiterConfig

from ..dataset import HarborTaskDataset
from ..tito_harbor_generator import TITOHarborGenerator


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)

    # Eval-time context-budget override. When set, eval trajectories (batch_metadata
    # .training_phase == "eval") use this as model_info.max_input_tokens/max_output_tokens
    # instead of the train-time value, letting eval run at a longer context than training.
    # Requires the vLLM engine's max_model_len to be >= this value.
    eval_max_model_len: Optional[int] = None

    # Per-instance masking policy for failed trajectories
    # (see TITOHarborGenerator._mask_failed_instances_and_compute_metrics):
    #   If an instance has fewer than `timeout_mask_instance_threshold`
    #   agent_timeout trajectories, only the timed-out trajectories are
    #   loss-masked; the rest of the instance is kept for training. At or
    #   above the threshold, the whole instance is masked (same as the
    #   conservative behavior for non-timeout errors).
    timeout_mask_instance_threshold: int = 2


@dataclass
class TITOHarborConfig(SkyRLTrainConfig):
    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    harbor_trial_config_file: str = "archipelago_tito"
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


# Environment variables to forward from head node to Ray workers
HARBOR_ENV_VARS = [
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "FMP_API_KEY",
    "TERRAPIN_API_KEY",
    "HF_TOKEN",
    "GOOGLE_API_KEY",
    "REDUCTO_API_KEY",
]


def initialize_ray_with_harbor_env(cfg) -> None:
    """Import-only replacement for skyrl.train.utils.utils.initialize_ray.

    Mirrors upstream (prepare_runtime_environment -> ray.init -> sync_registries)
    but additionally forwards harbor/Modal credentials and the CUDA toolkit
    location into the job-level Ray runtime env, so trial tasks and Megatron
    workers see them. Upstream's initialize_ray has no hook for extra env
    vars, so the recipe owns this wrapper instead.

    CUDA_HOME matters for TileLang's GDN kernel JIT (Qwen3.5/3.6): without it,
    tilelang falls back to the pip cu13 nvcc which ships no CCCL headers and the
    GDN backward dies with `fatal error: cuda/atomic: No such file or directory`.
    """
    from datetime import datetime

    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries
    from skyrl.train.utils.utils import prepare_runtime_environment

    verbose_logging = os.environ.get("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "0") == "1"
    if not verbose_logging:
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"

    env_vars = prepare_runtime_environment(cfg)

    for var_name in HARBOR_ENV_VARS + ["CUDA_HOME", "CUDA_PATH"]:
        if value := os.environ.get(var_name):
            env_vars[var_name] = value

    if not verbose_logging:
        log_path = Path(cfg.trainer.log_path).resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = str(log_path / f"infra-{timestamp}.log")
        os.environ["SKYRL_LOG_FILE"] = log_file
        env_vars["SKYRL_LOG_FILE"] = log_file

    ray.init(runtime_env={"env_vars": env_vars}, log_to_driver=True)

    if not verbose_logging:
        logger.info(f"Infrastructure logs will be written to: {log_file}")

    sync_registries()


class TITOHarborExp(BasePPOExp):
    """Base experiment class for TITO Harbor eval and training."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return TITOHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        if self.cfg.data.val_data:
            return HarborTaskDataset(data_files=self.cfg.data.val_data)
        return None


def load_config_and_launch(entrypoint_fn):
    """Common main() logic: parse config, load YAML, forward env vars, launch Ray."""
    cfg = TITOHarborConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor trial config from YAML file and merge CLI overrides
    config_dir = Path(__file__).parent.parent / "harbor_trial_config"
    config_file = config_dir / f"{cfg.harbor_trial_config_file}.yaml"
    if config_file.exists():
        with open(config_file) as f:
            defaults = yaml.safe_load(f)
        cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
        logger.info(f"Loaded harbor trial config from: {config_file}")
    else:
        logger.warning(f"Harbor trial config file not found: {config_file}")

    validate_cfg(cfg)

    # Collect env vars from head node to forward to workers
    harbor_env_vars = {k: os.environ[k] for k in HARBOR_ENV_VARS if k in os.environ}
    logger.info(f"Forwarding {len(harbor_env_vars)} env vars to workers: {list(harbor_env_vars.keys())}")

    initialize_ray_with_harbor_env(cfg)

    pin_ip = os.environ.get("SKYRL_PIN_NODE_IP")
    if pin_ip:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        nodes = ray.nodes()
        node_ids = [n["NodeID"] for n in nodes if n.get("NodeManagerAddress") == pin_ip and n.get("Alive")]
        if not node_ids:
            raise ValueError(
                f"SKYRL_PIN_NODE_IP={pin_ip} not found among alive Ray nodes: "
                f"{[n['NodeManagerAddress'] for n in nodes if n.get('Alive')]}"
            )
        logger.info(f"Pinning entrypoint to node {pin_ip} (NodeID={node_ids[0][:12]}...)")
        sched = NodeAffinitySchedulingStrategy(node_id=node_ids[0], soft=False)
        ray.get(entrypoint_fn.options(scheduling_strategy=sched).remote(cfg, harbor_env_vars))
    else:
        ray.get(entrypoint_fn.remote(cfg, harbor_env_vars))
