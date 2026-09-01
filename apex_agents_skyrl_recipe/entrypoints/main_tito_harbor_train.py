"""
Training entrypoint for TITO Harbor tasks.

Uses TITOHarborGenerator for RL training with pre-computed token IDs,
loss masks, and logprobs from the archipelago agent's TITO state.

Usage:
    uv run --isolated --extra megatron --extra harbor -m \
        apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_train \
        trainer.policy.model.path=zai-org/GLM-4.7-Flash ...
"""

import os

import ray

from .common import TITOHarborExp, load_config_and_launch


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg, harbor_env_vars):
    for k, v in harbor_env_vars.items():
        os.environ[k] = v
    exp = TITOHarborExp(cfg)
    exp.run()


def main() -> None:
    load_config_and_launch(skyrl_entrypoint)


if __name__ == "__main__":
    main()
