# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)





def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }




def compute_normalized_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalized forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise normalized KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )

    # original masses before normalization, used for metrics
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    # ===== minimal change starts here =====
    # Normalize both teacher and student distributions on teacher top-k support.
    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)

    student_topk_log_probs = student_topk_log_probs - student_log_mass
    teacher_topk_log_probs = teacher_topk_log_probs - teacher_log_mass

    distillation_losses = kl_divergence(
        log_q=student_topk_log_probs,
        log_p=teacher_topk_log_probs,
    )
    # ===== minimal change ends here =====

    # Diagnostics for tracking teacher/student top-k overlap in OPD.
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    token_kl = teacher_topk_log_probs.exp() * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_my_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute collapsed forward KL distillation loss using teacher top-k + other token.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. student full log probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # 3. student top-k ids only for overlap diagnostics
    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # 4. gather student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )

    # loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    # ============================================================
    # My method: teacher top-k + one collapsed "other" token
    # ============================================================

    # top-k probabilities
    teacher_topk_probs = teacher_topk_log_probs.exp()
    student_topk_probs = student_topk_log_probs.exp()

    # mass on teacher top-k support
    teacher_mass = teacher_topk_probs.sum(dim=-1)  # (bsz, seqlen/sp_size)
    student_mass = student_topk_probs.sum(dim=-1)  # (bsz, seqlen/sp_size)

    # collapsed other probability
    teacher_other_prob = (1.0 - teacher_mass)
    student_other_prob = (1.0 - student_mass)

    teacher_other_log_prob = teacher_other_prob.log().unsqueeze(-1)
    student_other_log_prob = student_other_prob.log().unsqueeze(-1)

    # append other as the last pseudo-token
    teacher_collapsed_log_probs = torch.cat(
        [teacher_topk_log_probs, teacher_other_log_prob],
        dim=-1,
    )

    student_collapsed_log_probs = torch.cat(
        [student_topk_log_probs, student_other_log_prob],
        dim=-1,
    )

    # KL(teacher_collapsed || student_collapsed)
    distillation_losses = kl_divergence(
        log_q=student_collapsed_log_probs,
        log_p=teacher_collapsed_log_probs,
    )

    # ============================================================
    # Diagnostics for teacher/student top-k overlap
    # ============================================================

    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    # only use real teacher top-k tokens here, not the collapsed other token
    token_kl = teacher_topk_probs * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }





def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_p=student_topk_log_probs, log_q=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }




def compute_normalized_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalized forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise normalized KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )

    # original masses before normalization, used for metrics
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    # ===== minimal change starts here =====
    # Normalize both teacher and student distributions on teacher top-k support.
    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)

    student_topk_log_probs = student_topk_log_probs - student_log_mass
    teacher_topk_log_probs = teacher_topk_log_probs - teacher_log_mass

    distillation_losses = kl_divergence(
        log_p=student_topk_log_probs,
        log_q=teacher_topk_log_probs,
    )
    # ===== minimal change ends here =====

    # Diagnostics for tracking teacher/student top-k overlap in OPD.
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    token_kl = teacher_topk_log_probs.exp() * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


import math
import torch
import torch.nn.functional as F

_LOG2 = math.log(2.0)


def log1mexp(log_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """数值稳定地计算 log(1 - exp(log_x))，要求 log_x <= 0。

    Mächler (2012) 两分支公式 + double-where，保证前向和反向都不出 inf/NaN。
    """
    log_x = log_x.clamp_max(-eps)  # 兜底 mass==1 (-inf) 与 mass>1 (log负数) 两种病态
    cond = log_x > -_LOG2
    a_in = torch.where(cond, log_x, torch.full_like(log_x, -_LOG2))
    b_in = torch.where(cond, torch.full_like(log_x, -_LOG2), log_x)
    out_a = torch.log(-torch.expm1(a_in))   # (-log2, 0]
    out_b = torch.log1p(-torch.exp(b_in))   # (-inf, -log2]
    return torch.where(cond, out_a, out_b)


def compute_my_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute collapsed reverse KL distillation loss using teacher top-k + collapsed
    "other" token. Numerically stable version (log-space, no 1 - sum(exp)).

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. student full log probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # 3. student top-k ids only for overlap diagnostics
    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # 4. gather student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )

    loss_config: DistillationLossConfig = config.distillation_loss

    # ============================================================
    # My method: teacher top-k + one collapsed "other" token (stable)
    # ============================================================

    # log 空间的 top-k 质量，用 logsumexp，绝不 exp-求和-再相减
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1)  # (1, nnz)
    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1)

    # 稳定地算 log(1 - mass)，结果天然有限（被 clamp_max(-eps) 兜底）
    teacher_other_log_prob = log1mexp(teacher_log_mass).unsqueeze(-1)
    student_other_log_prob = log1mexp(student_log_mass).unsqueeze(-1)

    # 线性 mass 仅供 metrics，含义与原实现一致
    teacher_mass = teacher_log_mass.exp()  # (bsz, seqlen/sp_size)
    student_mass = student_log_mass.exp()  # (bsz, seqlen/sp_size)

    # 拼上 other 伪 token
    teacher_collapsed_log_probs = torch.cat(
        [teacher_topk_log_probs, teacher_other_log_prob],
        dim=-1,
    )
    student_collapsed_log_probs = torch.cat(
        [student_topk_log_probs, student_other_log_prob],
        dim=-1,
    )

    # KL(student_collapsed || teacher_collapsed)
    distillation_losses = kl_divergence(
        log_p=student_collapsed_log_probs,
        log_q=teacher_collapsed_log_probs,
    )

    # ============================================================
    # Diagnostics for teacher/student top-k overlap
    # ============================================================

    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    # only use real teacher top-k tokens here, not the collapsed other token
    token_kl = teacher_topk_log_probs.exp() * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }



# def compute_my_reverse_kl_topk(
#     student_logits: torch.Tensor,
#     teacher_topk_log_probs: torch.Tensor,
#     teacher_topk_ids: torch.Tensor,
#     config: DistillationConfig,
#     data_format: str,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Compute collapsed reverse KL distillation loss using teacher top-k + other token.
#
#     Args:
#         student_logits: (bsz, seqlen/sp_size, vocab_size).
#         teacher_topk_log_probs: (bsz, seqlen, topk).
#         teacher_topk_ids: (bsz, seqlen, topk).
#         data_format: "thd" or "bshd".
#
#     Returns:
#     - distillation_losses: (bsz, seqlen/sp_size)
#     - student_mass: (bsz, seqlen/sp_size)
#     - teacher_mass: (bsz, seqlen/sp_size)
#     """
#     assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
#     teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
#     teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)
#
#     # 1. split across sp groups
#     if get_ulysses_sequence_parallel_world_size() > 1:
#         teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
#         teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
#
#     assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]
#
#     # 2. student full log probs
#     student_log_probs = F.log_softmax(student_logits, dim=-1)
#
#     # 3. student top-k ids only for overlap diagnostics
#     student_topk_ids = torch.topk(
#         student_log_probs,
#         k=teacher_topk_ids.shape[-1],
#         dim=-1,
#     ).indices
#
#     # 4. gather student probs on teacher top-k support
#     student_topk_log_probs = torch.gather(
#         student_log_probs,
#         dim=-1,
#         index=teacher_topk_ids,
#     )
#
#     loss_config: DistillationLossConfig = config.distillation_loss
#     # if loss_config.log_prob_min_clamp is not None:
#     #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
#     #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
#
#     # ============================================================
#     # My method: teacher top-k + one collapsed "other" token
#     # ============================================================
#
#     # top-k probabilities
#     teacher_topk_probs = teacher_topk_log_probs.exp()
#     student_topk_probs = student_topk_log_probs.exp()
#
#     # mass on teacher top-k support
#     teacher_mass = teacher_topk_probs.sum(dim=-1)  # (bsz, seqlen/sp_size)
#     student_mass = student_topk_probs.sum(dim=-1)  # (bsz, seqlen/sp_size)
#
#     # collapsed other probability
#     teacher_other_prob = (1.0 - teacher_mass)
#     student_other_prob = (1.0 - student_mass)
#
#     teacher_other_log_prob = teacher_other_prob.log().unsqueeze(-1)
#     student_other_log_prob = student_other_prob.log().unsqueeze(-1)
#
#     # append other as the last pseudo-token
#     teacher_collapsed_log_probs = torch.cat(
#         [teacher_topk_log_probs, teacher_other_log_prob],
#         dim=-1,
#     )
#
#     student_collapsed_log_probs = torch.cat(
#         [student_topk_log_probs, student_other_log_prob],
#         dim=-1,
#     )
#
#     # KL(teacher_collapsed || student_collapsed)
#     distillation_losses = kl_divergence(
#         log_p=student_collapsed_log_probs,
#         log_q=teacher_collapsed_log_probs,
#     )
#
#     # ============================================================
#     # Diagnostics for teacher/student top-k overlap
#     # ============================================================
#
#     overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
#     overlap_count = overlap_mask.sum(dim=-1)
#
#     # only use real teacher top-k tokens here, not the collapsed other token
#     token_kl = teacher_topk_probs * (
#         teacher_topk_log_probs - student_topk_log_probs
#     )
#
#     overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
#     overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
#     overlap_token_advantage = torch.where(
#         overlap_count > 0,
#         overlap_token_advantage,
#         torch.zeros_like(overlap_token_advantage),
#     )
#
#     return {
#         "distillation_losses": distillation_losses,
#         "student_mass": student_mass,
#         "teacher_mass": teacher_mass,
#         "overlap_count": overlap_count,
#         "overlap_token_advantage": overlap_token_advantage,
#     }