"""
Fully async training entrypoint for TITO Harbor tasks.

Uses FullyAsyncRayPPOTrainer for in-flight weight update training with
TITOHarborGenerator for pre-computed token IDs, loss masks, and logprobs.

Usage:
    uv run --isolated --extra megatron --extra harbor -m \
        apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_fully_async \
        trainer.policy.model.path=zai-org/GLM-4.7-Flash ...
"""

import os

import ray
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

from .common import TITOHarborExp, load_config_and_launch


class TITOHarborFullyAsyncExp(TITOHarborExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg, harbor_env_vars):
    for k, v in harbor_env_vars.items():
        os.environ[k] = v
    exp = TITOHarborFullyAsyncExp(cfg)
    exp.run()


def main() -> None:
    load_config_and_launch(skyrl_entrypoint)


if __name__ == "__main__":
    main()
