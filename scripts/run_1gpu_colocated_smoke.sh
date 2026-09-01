set -ex

# ---------------------------------------------------------------------------
# 1-GPU E2E smoke test of the apex recipe on OSS SkyRL + OSS harbor.
#
# Exercises the full loop on a single GPU (vLLM and Megatron share the device
# via colocate_all + CUDA-IPC weight sync): dataset -> harbor trial on Modal ->
# TITO rollout -> DPPO update -> checkpoint. Env forwarding (Modal creds,
# CUDA_HOME) happens inside the recipe's initialize_ray_with_harbor_env.
#
# Defaults to Qwen/Qwen3.5-0.8B for the fastest possible plumbing check.
# MODEL_NAME=Qwen/Qwen3.5-9B for the bigger dense model (needs ~19GB HBM).
#
# Run from the repo root:
#   bash scripts/run_1gpu_colocated_smoke.sh &> /mnt/local_storage/recipe_smoke.log
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."

if [ -d /opt/amazon/efa/lib ]; then
  export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
  export SKYRL_LD_LIBRARY_PATH_EXPORT=1
  export FI_PROVIDER=efa
  export FI_EFA_USE_DEVICE_RDMA=1
fi
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

# TileLang GDN JIT needs the full CUDA toolkit (pip cu13 nvcc lacks CCCL headers).
# Forwarded to Ray workers by initialize_ray_with_harbor_env.
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

export SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}"
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename $MODEL_NAME)}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-True}"

RUN_NAME="${RUN_NAME:-recipe_smoke_$(basename $MODEL_NAME | tr 'A-Z.' 'a-z_')_1gpu}"
MODAL_APP_NAME="${MODAL_APP_NAME:-apex_agents_skyrl_recipe_smoke}"
SAVE_DIR="${SAVE_DIR:-/mnt/local_storage}"
TRIALS_DIR="${TRIALS_DIR:-$SAVE_DIR/$RUN_NAME/trials_run_$(date +%m%d_%H%M)}"
EXPORT_PATH="${EXPORT_PATH:-$SAVE_DIR/$RUN_NAME/export}"
CKPT_PATH="${CKPT_PATH:-$SAVE_DIR/$RUN_NAME/ckpt}"
export SKYRL_CHECKPOINT_TMPDIR="${SKYRL_CHECKPOINT_TMPDIR:-$SAVE_DIR/ckpt_staging}"
mkdir -p "$TRIALS_DIR" "$EXPORT_PATH" "$CKPT_PATH" "$SKYRL_CHECKPOINT_TMPDIR" 2>/dev/null || true

DATA_DIR="${DATA_DIR:-/mnt/local_storage/data/harbor}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$DATA_DIR/apex-agents-dev-1928-tiny}"
TRAIN_DATA="['$TRAIN_DATA_DIR']"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-1}"
INFERENCE_TP="${INFERENCE_TP:-1}"
MEGATRON_TP="${MEGATRON_TP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_EP="${MEGATRON_EP:-1}"
MEGATRON_ETP="${MEGATRON_ETP:-null}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-16384}"
LLM_MAX_PER_TURN_TOKENS="${LLM_MAX_PER_TURN_TOKENS:-4096}"
LLM_CALL_TIMEOUT="${LLM_CALL_TIMEOUT:-1800}"
TOOL_RESULT_MAX_CHARS="${TOOL_RESULT_MAX_CHARS:-8000}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
TRAJECTORIES_PER_SECOND="${TRAJECTORIES_PER_SECOND:-3}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-2}"

POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-dppo}"
TEMPERATURE="${TEMPERATURE:-1.0}"

export DEFAULT_EXTRA_PROMPT='Note: you are a text-only agent and cannot view images. Image-returning tools (e.g. `*_read_image`, `pdfs_read_page_as_image`) return nothing usable - do not call them; use text-extraction tools instead. If information is only available as an image, state that it is inaccessible rather than guessing.'
export EXTRA_PROMPT="${EXTRA_PROMPT:-$DEFAULT_EXTRA_PROMPT}"

# Source secrets here (NOT via `uv run --env-file`): Ray's uv runtime-env hook
# re-runs the exact `uv run` args inside the extracted working_dir on every
# worker, where .env.apex does not exist (it is gitignored and therefore
# excluded from Ray's working_dir upload). The recipe's Ray init forwards the
# needed vars to workers via the job runtime env instead.
set -a; source .env.apex; set +a

uv run \
  -m apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_train \
  data.train_data="$TRAIN_DATA" \
  data.val_data="[]" \
  harbor_trial_config_file=archipelago_tito \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.agent.kwargs.tito_tokenizer_name=$MODEL_NAME \
  harbor_trial_config.agent.kwargs.tito_tool_call_parser=$TOOL_CALL_PARSER \
  harbor_trial_config.agent.kwargs.tito_reasoning_parser=$REASONING_PARSER \
  harbor_trial_config.agent.kwargs.tool_result_max_chars=$TOOL_RESULT_MAX_CHARS \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.llm_kwargs.temperature=$TEMPERATURE \
  harbor_trial_config.agent.kwargs.llm_kwargs.timeout=$LLM_CALL_TIMEOUT \
  harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens=$LLM_MAX_PER_TURN_TOKENS \
  "harbor_trial_config.agent.kwargs.extra_prompt='$EXTRA_PROMPT'" \
  harbor_trial_config.environment.kwargs.modal_app_name=$MODAL_APP_NAME \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.export_path=$EXPORT_PATH \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.strategy=megatron \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.distributed_executor_backend="mp" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE \
  trainer.algorithm.temperature=$TEMPERATURE \
  trainer.algorithm.loss_reduction=prompt_mean \
  trainer.algorithm.dppo.delta_low=0.15 \
  trainer.algorithm.dppo.delta_high=0.15 \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.eps_clip_high=4 \
  trainer.algorithm.eps_clip_low=0.5 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.zero_variance_filter_tol=1e-6 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  "trainer.policy.optimizer_config.adam_betas=[0.9,0.98]" \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.use_sample_packing=true \
  trainer.fused_lm_head_logprob=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.eval_interval=-1 \
  trainer.eval_before_train=false \
  trainer.ckpt_interval="${CKPT_INTERVAL:-1}" \
  trainer.max_ckpts_to_keep=1 \
  trainer.epochs=1 \
  trainer.resume_mode=none \
  trainer.logger="${LOGGER:-wandb}" \
  trainer.project_name="${PROJECT_NAME:-apex-recipe-smoke}" \
  trainer.run_name=$RUN_NAME \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_TP \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend="triton" \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=$TOOL_CALL_PARSER \
  generator.inference_engine.engine_init_kwargs.reasoning_parser=$REASONING_PARSER \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=false \
  generator.batched=false \
  generator.eval_max_model_len=$MAX_MODEL_LEN \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  "$@"
