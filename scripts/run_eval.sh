set -ex

# ---------------------------------------------------------------------------
# TITO eval on OSS SkyRL + OSS harbor (apex recipe).
# Runs the eval entrypoint over an eval task set. Defaults assume a ckpt or HF
# model at MODEL_PATH served by NUM_INFERENCE_ENGINES x INFERENCE_ENGINE_TP.
#
#   MODEL_NAME=Qwen/Qwen3.6-35B-A3B MODEL_PATH=/path/to/hf_ckpt \
#   EVAL_DATASET=apex-agents-eval-480-062926 \
#     bash scripts/run_eval.sh &> /mnt/local_storage/eval.log
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."

if [ -d /opt/amazon/efa/lib ]; then
  export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
  export SKYRL_LD_LIBRARY_PATH_EXPORT=1
  export FI_PROVIDER=efa
  export FI_EFA_USE_DEVICE_RDMA=1
fi
export NCCL_DEBUG=WARN
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export SKYRL_PIN_NODE_IP=${SKYRL_PIN_NODE_IP-}
export SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}

DATA_DIR="${DATA_DIR:-/mnt/local_storage/data/harbor}"
EVAL_DATASET="${EVAL_DATASET:-apex-agents-eval-480-062926}"
TRAIN_DATA="['$DATA_DIR/$EVAL_DATASET']"
EVAL_DATA="['$DATA_DIR/$EVAL_DATASET']"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-480}"

RUN_NAME="${RUN_NAME:-apex_agents_skyrl_recipe_eval}"
SAVE_DIR="${SAVE_DIR:-/mnt/local_storage}"
TRIALS_DIR="${TRIALS_DIR:-$SAVE_DIR/$RUN_NAME/eval_trials_run_$(date +%m%d_%H%M)}"
EXPORT_PATH="${EXPORT_PATH:-$SAVE_DIR/$RUN_NAME/export}"
CKPT_PATH="${CKPT_PATH:-$SAVE_DIR/$RUN_NAME/ckpt}"

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3.6-35B-A3B"}
MODEL_PATH="${MODEL_PATH:-$MODEL_NAME}"
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$(basename $MODEL_NAME)}
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
TRAJECTORIES_PER_SECOND="${TRAJECTORIES_PER_SECOND:-3}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-240}"
POLICY_NUM_NODES=${POLICY_NUM_NODES:-2}
POLICY_NUM_GPUS_PER_NODE=${POLICY_NUM_GPUS_PER_NODE:-8}
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-4}
INFERENCE_ENGINE_TP=${INFERENCE_ENGINE_TP:-4}
WEIGHT_TRANSFER_THRESHOLD_GB=${WEIGHT_TRANSFER_THRESHOLD_GB:-4.0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
TEMPERATURE=${TEMPERATURE:-1.0}
TIMEOUT_MASK_INSTANCE_THRESHOLD="${TIMEOUT_MASK_INSTANCE_THRESHOLD:-2}"
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-True}
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LLM_CALL_TIMEOUT="${LLM_CALL_TIMEOUT:-2000}"
MAX_STEPS="${MAX_STEPS:-250}"
TITO_BUDGET_WARNING_RATIO="${TITO_BUDGET_WARNING_RATIO:-null}"

export DEFAULT_EXTRA_PROMPT='Note: you are a text-only agent and cannot view images. Image-returning tools (e.g. `*_read_image`, `pdfs_read_page_as_image`) return nothing usable - do not call them; use text-extraction tools instead. If information is only available as an image, state that it is inaccessible rather than guessing.'
export EXTRA_PROMPT="${EXTRA_PROMPT:-$DEFAULT_EXTRA_PROMPT}"

# See run_1gpu_colocated_smoke.sh for why we source instead of uv --env-file.
set -a; source .env.apex; set +a

uv run \
  -m apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_eval \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  harbor_trial_config_file=archipelago_tito \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.agent.kwargs.tito_tokenizer_name=$MODEL_NAME \
  harbor_trial_config.agent.kwargs.tito_tool_call_parser=$TOOL_CALL_PARSER \
  harbor_trial_config.agent.kwargs.tito_reasoning_parser=$REASONING_PARSER \
  harbor_trial_config.agent.kwargs.max_steps=$MAX_STEPS \
  harbor_trial_config.agent.kwargs.tito_budget_warning_ratio=$TITO_BUDGET_WARNING_RATIO \
  harbor_trial_config.agent.kwargs.tool_result_max_chars=42000 \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=$MAX_MODEL_LEN \
  harbor_trial_config.agent.kwargs.llm_kwargs.temperature=$TEMPERATURE \
  harbor_trial_config.environment.kwargs.modal_app_name=${MODAL_APP_NAME:-$RUN_NAME} \
  harbor_trial_config.agent.kwargs.llm_kwargs.timeout=$LLM_CALL_TIMEOUT \
  "harbor_trial_config.agent.kwargs.extra_prompt='$EXTRA_PROMPT'" \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.export_path=$EXPORT_PATH \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.strategy=megatron \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$POLICY_NUM_GPUS_PER_NODE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.eval_batch_size=$EVAL_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.temperature=$TEMPERATURE \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.use_sample_packing=true \
  trainer.flash_attn=true \
  trainer.eval_interval=10 \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=${CKPT_INTERVAL:-999} \
  trainer.max_ckpts_to_keep=10 \
  trainer.epochs=2 \
  trainer.resume_mode=null \
  trainer.logger=${LOGGER:-wandb} \
  trainer.project_name=${PROJECT_NAME:-apex-recipe} \
  trainer.run_name=$RUN_NAME \
  generator.inference_engine.served_model_name=$SERVED_MODEL_NAME \
  generator.inference_engine.distributed_executor_backend="mp" \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend="triton" \
  generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB=$WEIGHT_TRANSFER_THRESHOLD_GB \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=$TOOL_CALL_PARSER \
  generator.inference_engine.engine_init_kwargs.reasoning_parser=$REASONING_PARSER \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=false \
  generator.timeout_mask_instance_threshold=$TIMEOUT_MASK_INSTANCE_THRESHOLD \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  "$@"
