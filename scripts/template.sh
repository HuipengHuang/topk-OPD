#!/bin/bash
#SBATCH --job-name=opd-distill-qwen
#SBATCH --partition=A100
#SBATCH --nodelist=gpu-08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x
set -e

# ========== 初始化 ==========
cd /data/home/huanghp/ppl_unsupervised_rl
mkdir -p logs

module load python/miniconda3/26.3.2

echo "==== Training OPD Distillation ===="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate unsupervised

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

# ========== 训练命令 ==========
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['/data/home/huanghp/data/gsm8k/train.parquet', '/data/home/huanghp/data/math/train.parquet']" \
  data.val_files="['/data/home/huanghp/data/gsm8k/test.parquet', '/data/home/huanghp/data/math/test.parquet']" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=False \
  actor_rollout_ref.model.path=/data/common/LLMs/Qwen/Qwen3-8B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.max_model_len=3073 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=4 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path=/data/common/LLMs/Qwen/Qwen3-32B \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=2 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
  distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.balance_batch=True \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name=verl_distill_gsm8k_math \
  trainer.experiment_name=qwen3_8b_from_qwen3_32b_k1 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=200 \
  trainer.test_freq=5 \
  trainer.total_epochs=15 \
  trainer.default_local_dir=/data/home/huanghp/ppl_unsupervised_rl/checkpoint/verl_distill_gsm8k_math/qwen3_8b_from_qwen3_32b_k1

echo "==== Done ===="