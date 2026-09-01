set -ex

# ---------------------------------------------------------------------------
# Qwen3.6-35B-A3B fully-async hero run (flattened, single script).
#
# The production configuration from the blogpost, baked in as defaults:
# DPPO + prompt_mean (deltas 0.15), 160k context, 10 engines x TP=8 +
# 4 training nodes (Megatron TP=8 EP=8), MAX_CONCURRENCY=550.
# Every knob stays ${VAR:-default}-overridable.
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/.."

# AWS EFA networking (only when EFA libs exist on the node)
if [ -d /opt/amazon/efa/lib ]; then
  export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
  export SKYRL_LD_LIBRARY_PATH_EXPORT=1
  export FI_PROVIDER=efa
  export FI_EFA_USE_DEVICE_RDMA=1
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset NCCL_NET 2>/dev/null || true
unset NCCL_NET_PLUGIN 2>/dev/null || true
unset NCCL_CONF_FILE 2>/dev/null || true

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

# First-launch timeouts (uv env build / TE JIT / vLLM warm-up on cold nodes)
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}
export SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}

# Entrypoint pin (harbor trials + skyrl_entrypoint live here; pick a GPU node
# with plenty of host RAM — harbor trials add ~370GB). Empty = unpinned.
export SKYRL_PIN_NODE_IP=${SKYRL_PIN_NODE_IP-}

export RUN_NAME=${RUN_NAME:-apex_qwen36_35b_tito_rl_dev1928}
MODAL_APP_NAME=${MODAL_APP_NAME:-apex_qwen36_35b}
export SAVE_DIR=${SAVE_DIR:-/mnt/local_storage}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-160000}
export EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-262144}

TRIALS_DIR="${TRIALS_DIR:-$SAVE_DIR/$RUN_NAME/trials_run_$(date +%m%d_%H%M)}"
EXPORT_PATH="${EXPORT_PATH:-$SAVE_DIR/$RUN_NAME/export}"
export SKYRL_CHECKPOINT_TMPDIR=${SKYRL_CHECKPOINT_TMPDIR:-$SAVE_DIR/ckpt_staging}
mkdir -p "$SKYRL_CHECKPOINT_TMPDIR" 2>/dev/null || true
export CKPT_PATH=${CKPT_PATH:-$SAVE_DIR/$RUN_NAME/ckpt}

DATA_DIR="${DATA_DIR:-/mnt/local_storage/data/harbor}"
TRAIN_DATA="['${TRAIN_DATA_DIR:-$DATA_DIR/apex-agents-dev-1928}']"
EVAL_DATA="['${EVAL_DATA_DIR:-$DATA_DIR/apex-agents-eval-480-062926}']"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}"

# RESUME: set RESUME_PATH to an s3:// or local checkpoint dir (e.g.
# .../ckpt/global_step_N) to resume from it. skyrl's io layer takes s3:// URIs
# directly and shells out to s5cmd — it must be on PATH on all nodes.
# Empty (default) starts fresh / falls back to resume_mode=latest.
RESUME_PATH="${RESUME_PATH-}"
RESUME_ARGS=""
if [ -n "$RESUME_PATH" ]; then
  RESUME_ARGS="trainer.resume_mode=from_path trainer.resume_path=$RESUME_PATH"
fi

#-----------------------
# Training parameters
#-----------------------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
POLICY_MINI_BATCH_SIZE="${POLICY_MINI_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_TOKENS_PER_MICROBATCH="${MAX_TOKENS_PER_MICROBATCH:-160000}"

#-----------------
# vLLM parameters (GPU-only KV cache: Qwen3.6 is a GDN hybrid — KV-offload
# connectors disable the hybrid KV manager and break EngineCore init)
#-----------------
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

#-----------------------
# Fully async knobs
#-----------------------
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-3}"
# max workers = mini_batch * (max_staleness + 1) = 16 * 4 = 64
NUM_PARALLEL_GENERATION_WORKERS="${NUM_PARALLEL_GENERATION_WORKERS:-64}"
CLEAR_KV_CACHE_ON_WEIGHT_SYNC="${CLEAR_KV_CACHE_ON_WEIGHT_SYNC:-false}"
SAMPLE_FULL_BATCH=${SAMPLE_FULL_BATCH:-true}

#-----------------------
# Infrastructure: 14 usable H100 nodes = 10 inference (10 engines x TP=8)
# + 4 training (Megatron TP=8, EP=8, CP=1, PP=1)
#-----------------------
INFERENCE_TP="${INFERENCE_TP:-8}"
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-10}"
POLICY_NUM_GPUS_PER_NODE="${POLICY_NUM_GPUS_PER_NODE:-8}"
POLICY_NUM_NODES="${POLICY_NUM_NODES:-4}"
MEGATRON_TP="${MEGATRON_TP:-8}"
MEGATRON_EP="${MEGATRON_EP:-8}"
MEGATRON_CP="${MEGATRON_CP:-1}"
MEGATRON_PP="${MEGATRON_PP:-1}"
MEGATRON_ETP="${MEGATRON_ETP:-1}"

LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-True}
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

#---------------
# Rate limiting (550 = 660 * 10/12, scaled with the engine count)
#---------------
ENABLE_RATE_LIMITING=${ENABLE_RATE_LIMITING:-true}
TRAJECTORIES_PER_SECOND="${TRAJECTORIES_PER_SECOND:-3}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-550}"

#------------------------
# Algorithm: DPPO (binary TV) + prompt_mean, GLM-5-paper clips
#------------------------
export POLICY_LOSS_TYPE=${POLICY_LOSS_TYPE:-dppo}
LOSS_REDUCTION="${LOSS_REDUCTION:-prompt_mean}"
DPPO_DELTA_LOW="${DPPO_DELTA_LOW:-0.15}"
DPPO_DELTA_HIGH="${DPPO_DELTA_HIGH:-0.15}"
GRPO_NORM_BY_STD=${GRPO_NORM_BY_STD:-false}
USE_KL_LOSS=${USE_KL_LOSS:-false}
ZERO_VARIANCE_FILTER=${ZERO_VARIANCE_FILTER:-true}
TEMPERATURE=${TEMPERATURE:-1.0}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-4}
EPS_CLIP_LOW=${EPS_CLIP_LOW:-0.5}

LR=${LR:-1.0e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
ADAM_BETAS="${ADAM_BETAS:-[0.9,0.98]}"

# -------
# Harness
# -------
LLM_MAX_PER_TURN_TOKENS="${LLM_MAX_PER_TURN_TOKENS:-50000}"
LLM_CALL_TIMEOUT="${LLM_CALL_TIMEOUT:-1800}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-3600}"

TOOL_RESULT_MAX_CHARS="${TOOL_RESULT_MAX_CHARS:-42000}"
TITO_BUDGET_WARNING_RATIO="${TITO_BUDGET_WARNING_RATIO:-0.2}"

# Directives: text-only note + code-exec usage + spreadsheet-formula
# and PDF-layout re-extraction guidance (post new-vs-old investigations).
export DEFAULT_EXTRA_PROMPT='Note: you are a text-only agent and cannot view images. Image-returning tools (e.g. `*_read_image`, `pdfs_read_page_as_image`) return nothing usable - do not call them; use text-extraction tools instead. If information is only available as an image, state that it is inaccessible rather than guessing. When you need to run code or calculations, call the `code_execution_code_exec` tool with your shell command in its `code` argument, and always write the `code` argument name explicitly on every call. IMPORTANT for SPREADSHEETS / financial models: the `excel_read_tab` tool shows only computed cell VALUES, not the underlying FORMULAS. When a question depends on the model'\'''\''s STRUCTURE or logic (which inputs/assumptions drive an output, dependency chains, sensitivities, how a figure is calculated) - not just reading a final number - locate the .xlsx file on disk (`filesystem_search_files`) and re-read it yourself with `code_execution_code_exec` using openpyxl: `wb = openpyxl.load_workbook(path, data_only=False)` then inspect `cell.value` to see formula strings (e.g. '\'''\''=K51/K52'\'''\''); also load with `data_only=True` for computed values. Use the formulas to trace how the model computes each output before answering. IMPORTANT for PDFs with CHARTS/TABLES/INFOGRAPHICS: the `pdfs_read_pdf_pages` tool often flattens the 2-D layout into an ambiguous bare number sequence (labels may be out of order), making charts and multi-column tables impossible to map reliably. When you hit such a page, DO NOT guess from the flattened text. Instead locate the actual PDF file on disk (`filesystem_search_files`) and re-extract it with `code_execution_code_exec` using a layout-preserving library: prefer `pdfplumber` (`page.extract_tables()` for tables, or `page.extract_text(layout=True)` for column alignment) or `camelot`/`tabula`. Reconstruct the table from that layout-preserved output and cross-check against any narrative figure before computing your answer. Most Python packages you need are ALREADY INSTALLED in the code sandbox (pandas, numpy, openpyxl, python-docx, PyPDF2, pypdf, pdfplumber, pymupdf/fitz, python-pptx, xlrd, xlsxwriter, pillow, pytesseract). `pip install` DOES NOT WORK in this sandbox — it fails with a permission error (HOME is on a read-only mount). If you really have to, you could use `pip install --target=` to redirect.'
export EXTRA_PROMPT="${EXTRA_PROMPT:-$DEFAULT_EXTRA_PROMPT}"

# Source secrets (NOT via `uv run --env-file` — Ray's uv hook re-runs the exact
# uv args in the working_dir extract, where the gitignored file doesn't exist).
set -a; source .env.apex; set +a

uv run \
  -m apex_agents_skyrl_recipe.entrypoints.main_tito_harbor_fully_async \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  harbor_trial_config_file=archipelago_tito \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.agent.override_timeout_sec=$AGENT_TIMEOUT \
  harbor_trial_config.agent.kwargs.tito_tokenizer_name=$MODEL_NAME \
  harbor_trial_config.agent.kwargs.tito_tool_call_parser=$TOOL_CALL_PARSER \
  harbor_trial_config.agent.kwargs.tito_reasoning_parser=$REASONING_PARSER \
  harbor_trial_config.agent.kwargs.tito_budget_warning_ratio=$TITO_BUDGET_WARNING_RATIO \
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
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$POLICY_NUM_GPUS_PER_NODE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=$MAX_STALENESS_STEPS \
  trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_GENERATION_WORKERS \
  trainer.fully_async.clear_kv_cache_on_weight_sync=$CLEAR_KV_CACHE_ON_WEIGHT_SYNC \
  trainer.fully_async.sample_full_batch=$SAMPLE_FULL_BATCH \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS_TYPE \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.dppo.delta_low=$DPPO_DELTA_LOW \
  trainer.algorithm.dppo.delta_high=$DPPO_DELTA_HIGH \
  trainer.algorithm.temperature=$TEMPERATURE \
  trainer.algorithm.grpo_norm_by_std=$GRPO_NORM_BY_STD \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  trainer.algorithm.zero_variance_filter=$ZERO_VARIANCE_FILTER \
  trainer.algorithm.zero_variance_filter_tol=1e-6 \
  trainer.policy.optimizer_config.lr=$LR \
  "trainer.policy.optimizer_config.adam_betas=$ADAM_BETAS" \
  trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY \
  trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM \
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
  trainer.fused_lm_head_logprob=${FUSED_LM_HEAD_LOGPROB:-true} \
  trainer.eval_interval=${EVAL_INTERVAL:-20} \
  trainer.eval_before_train=${EVAL_BEFORE_TRAIN:-False} \
  trainer.ckpt_interval=${CKPT_INTERVAL:-5} \
  trainer.max_ckpts_to_keep=${MAX_CKPTS_TO_KEEP:-5} \
  trainer.epochs=${EPOCHS:-3} \
  trainer.resume_mode=${RESUME_MODE:-latest} \
  trainer.logger=${LOGGER:-wandb} \
  trainer.project_name=${PROJECT_NAME:-apexagents-rl} \
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
  generator.inference_engine.engine_init_kwargs.max_model_len=$EVAL_MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend="triton" \
  generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=true \
  generator.inference_engine.engine_init_kwargs.tool_call_parser=$TOOL_CALL_PARSER \
  generator.inference_engine.engine_init_kwargs.reasoning_parser=$REASONING_PARSER \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.apply_overlong_filtering=false \
  generator.batched=false \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  generator.eval_max_model_len=$EVAL_MAX_MODEL_LEN \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  generator.inference_engine.router_init_kwargs.policy=sticky_least_loaded \
  $RESUME_ARGS \
  "$@"
