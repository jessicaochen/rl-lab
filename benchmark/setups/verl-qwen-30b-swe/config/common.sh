#!/usr/bin/env bash
# Shared launcher for the scalability ladder (h200-t<T>-s<S> variants).
# Every topology/batch knob is a plain shell variable set by the variant's
# run.sh before it exec's this file. Convention: ${NAME} / ${NAME:-d} are
# rlbench RENDER-time variables (baked into the run folder copy); bare $name
# is shell-time and never touched by rlbench.
set -euo pipefail

MODEL_PATH="Qwen/Qwen3-30B-A3B-Thinking-2507"
DATA_DIR=/data/datasets/r2e_gym-4096   # shared, idempotent
CKPTS_DIR=/data/outputs/${RUN_ID:-run}/checkpoints
AGENT_LOG_DIR=/data/outputs/${RUN_ID:-run}/agent-logs
mkdir -p "$DATA_DIR" "$AGENT_LOG_DIR"

# dataset: small fixed slice, idempotent
if [ ! -f "$DATA_DIR/r2e_gym.parquet" ]; then
  python3 -m rlbench_verl_tasks.r2e_gym.preprocess \
    --local-save-dir "$DATA_DIR" --max-instances 4096
fi

RUN_OUT=/data/outputs/${RUN_ID:-run}

# validation: SWE-bench Verified (held-out; never trained on). Full 500 are
# preprocessed once by uni-agent's swe_bench task; a deterministic subset
# (sorted by instance_id, first VAL_TASKS) is what each run validates on —
# 100 by default, `--var VAL_TASKS=500` for the full set, `--var VAL_TASKS=0`
# turns validation off (pure step-timing runs; verl still needs *a* val file,
# so it is pointed at the training slice and never run).
if [ "$VAL_TASKS" -gt 0 ]; then
  VAL_DIR=/data/datasets/swebench_verified
  if [ ! -f "$VAL_DIR/swe_bench_verified.parquet" ]; then
    python3 -m uni_agent.tasks.swe_bench.preprocess --local-save-dir "$VAL_DIR"
  fi
  VAL_FILE="$VAL_DIR/swe_bench_verified-first$VAL_TASKS.parquet"
  if [ ! -f "$VAL_FILE" ]; then
    python3 - "$VAL_DIR/swe_bench_verified.parquet" "$VAL_FILE" "$VAL_TASKS" <<'PY'
import sys
from datasets import load_dataset
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = load_dataset("parquet", data_files=src, split="train")
ids = [r["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"] for r in ds]
order = sorted(range(len(ids)), key=lambda i: ids[i])[:n]
ds.select(order).to_parquet(dst)
print(f"wrote {len(order)} validation instances -> {dst}")
PY
  fi
  VAL_ARGS=(data.val_batch_size="$VAL_TASKS" trainer.val_before_train=True trainer.test_freq="$TOTAL_STEPS")
else
  VAL_FILE="$DATA_DIR/r2e_gym.parquet"
  VAL_ARGS=(data.val_batch_size=null trainer.val_before_train=False trainer.test_freq=-1)
fi

# extra Ray runtime-env variables for every actor (trainer ranks, vLLM
# servers, weight-sync engine): `--var 'RAY_ENV_VARS=K=V K2=V2'` (space
# separated; values may contain commas). Used by the repo-level experiments/ to flip NCCL
# transports per run without touching the RayCluster pods.
EXTRA_OVERRIDES=()
for kv in ${RAY_ENV_VARS:-}; do
  EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.${kv%%=*}=\"${kv#*=}\"")
done

# pre-download the model once (head) so workers only ever read the shared
# cache: concurrent cross-node downloads onto Filestore cause NFS stale file
# handles mid-load
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-30B-A3B-Thinking-2507")
print("model snapshot complete")
PY

PROMPT_LENGTH=8192
RESPONSE_LENGTH=32768
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
# --- knobs every variant run.sh must set ---------------------------------
for v in TRAIN_NNODES TRAIN_GPUS TRAIN_TP TRAIN_CP TRAIN_EP ROLLOUT_GPUS ROLLOUT_TP \
         TRAIN_BATCH ROLLOUT_N SYNC_STEP MINI_BATCH SESSIONS GATEWAYS AGENT_WORKERS TOTAL_STEPS VAL_TASKS; do
  eval "test -n \"\${$v:-}\"" || { echo "common.sh: $v must be set by the variant run.sh" >&2; exit 2; }
done
if [ $((SYNC_STEP * MINI_BATCH)) -ne "$TRAIN_BATCH" ]; then
  echo "common.sh: separate_async requires TRAIN_BATCH == SYNC_STEP * MINI_BATCH ($TRAIN_BATCH != $SYNC_STEP*$MINI_BATCH)" >&2; exit 2
fi
# Keep TP*CP*EP <= 8 so the expert-parallel group stays on one NVLink island
# (cross-node EP all-to-all over pod TCP measured ~60 Gbps / 78-min steps).
if [ $((TRAIN_TP * TRAIN_CP * TRAIN_EP)) -gt 8 ]; then
  echo "common.sh: TRAIN_TP*TRAIN_CP*TRAIN_EP > 8 spans nodes; refusing without RDMA" >&2; exit 2
fi
ACTOR_PPO_MAX_TOKEN_LEN=$(( MAX_MODEL_LEN / TRAIN_CP ))

exec python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    "hydra.searchpath=[pkg://verl.trainer.config]" \
    +ray_kwargs.ray_init.address=auto \
    '+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=""' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="eth0"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="/data/hf-cache"' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"' \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step="$SYNC_STEP" \
    trainer.v1.sampler.max_off_policy_threshold=16 \
    transfer_queue.enable=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    data.train_files="['$DATA_DIR/r2e_gym.parquet']" \
    data.val_files="['$VAL_FILE']" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size="$TRAIN_BATCH" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH \
    actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_GPUS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    +actor_rollout_ref.rollout.custom.agent_framework.gateway_count="$GATEWAYS" \
    "+actor_rollout_ref.rollout.custom.agent_framework.log_dir=$AGENT_LOG_DIR" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="$SESSIONS" \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=2400 \
    +actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=task_config.yaml \
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=$MODEL_PATH" \
    +actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=True \
    actor_rollout_ref.actor.checkpoint.strict=False \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=$TRAIN_CP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=$TRAIN_CP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$TRAIN_EP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    trainer.project_name=rlbench-verl \
    trainer.experiment_name="${RUN_ID:-smoke}" \
    'trainer.logger=["console"]' \
    trainer.save_freq=-1 \
    trainer.rollout_data_dir="$RUN_OUT/rollouts" \
    +trainer.validation_data_dir="$RUN_OUT/val-rollouts" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir=/ckpt-gcs/${RUN_ID} \
    trainer.nnodes="$TRAIN_NNODES" \
    trainer.n_gpus_per_node="$TRAIN_GPUS" \
    "${VAL_ARGS[@]}" \
    "${EXTRA_OVERRIDES[@]}"
