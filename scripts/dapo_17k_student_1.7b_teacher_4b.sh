#!/bin/bash
#SBATCH --job-name=opd-distill-qwen
#SBATCH --partition=P6000
#SBATCH --nodelist=gpu-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:5
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x
set -e

# ========== 初始化 ==========
cd /data/home/huanghp/topk-opd
mkdir -p logs

module load python/miniconda3/26.3.2

echo "==== Training OPD Distillation ===="
source /data/softwares/miniconda3/26.3.2-2/etc/profile.d/conda.sh
conda activate opd

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

# ========== 训练参数 ==========
STUDENT_MODEL=/data/common/LLMs/Qwen3-1.7B-Base
TEACHER_MODEL=/data/common/LLMs/Qwen3-4B

TRAIN_DATA=/data/home/huanghp/data/DAPO-Math-17k-Processed/dapo-math-17k-processed.parquet
VAL_DATA=/data/home/huanghp/data/OpenR1-Math-46k/valid_math500.parquet
CHECKPOINT_DIR=/data/home/huanghp/topk-opd/checkpoint

LR=1e-6
N_ROLLOUTS=4
DISTILL_TOPK=8
DISTILL_LOSS_MODE=normalized_reverse_kl_topk

PROJECT_NAME=dapo17k_qwen3_17b_base_from_qwen3_4b
EXPERIMENT_NAME=loss_${DISTILL_LOSS_MODE}_topk_${DISTILL_TOPK}_lr_${LR}

# ========== 训练命令 ==========
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${TRAIN_DATA}']" \
  data.val_files="['${VAL_DATA}']" \
  data.train_batch_size=72 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  \
  reward.reward_manager.name=remote \
  critic.enable=False \
  \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.path=${STUDENT_MODEL} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=72 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL} \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.max_model_len=4097 \
  distillation.distillation_loss.loss_mode=${DISTILL_LOSS_MODE} \
  distillation.distillation_loss.topk=${DISTILL_TOPK} \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  trainer.balance_batch=True \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.save_freq=500 \
  trainer.total_training_steps=500 \
  trainer.test_freq=5 \
  trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}

echo "==== Done ===="

#