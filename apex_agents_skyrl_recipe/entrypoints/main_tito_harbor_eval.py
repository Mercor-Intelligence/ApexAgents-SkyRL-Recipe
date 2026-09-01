"""
Eval-only entrypoint for TITO Harbor tasks.

Launches vLLM inference engines via Ray, runs the TITOHarborGenerator on all
tasks in the dataset, and reports pass@1 and average reward.

Usage:
    uv run --isolated --extra megatron --extra harbor -m \
        apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_eval \
        trainer.policy.model.path=zai-org/GLM-4.7-Flash ...
"""

import asyncio
import os

import ray
from loguru import logger
from skyrl.train.evaluate import evaluate
from skyrl.train.utils.trainer_utils import build_dataloader

from .common import TITOHarborExp, load_config_and_launch


class TITOHarborEvalExp(TITOHarborExp):

    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run_eval(self):
        """Run eval-only: generate trajectories and report metrics."""
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        inference_engine_client = self.get_inference_client()
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        eval_dataloader = build_dataloader(self.cfg, self.eval_dataset, is_train=False)

        logger.info(f"Evaluating {len(self.eval_dataset)} tasks (pass@1)")

        eval_metrics = await evaluate(
            eval_dataloader=eval_dataloader,
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        logger.info(f"Evaluation metrics: {eval_metrics}")

        return eval_metrics


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg, harbor_env_vars):
    for k, v in harbor_env_vars.items():
        os.environ[k] = v
    exp = TITOHarborEvalExp(cfg)
    return asyncio.run(exp.run_eval())


def main() -> None:
    load_config_and_launch(skyrl_entrypoint)


if __name__ == "__main__":
    main()
