"""Torch batch scorer for Phase 3c GPU-first training."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from alpha_etf.gpt.evaluation import FormulaScoreConfig
from alpha_etf.gpt.torch_vm import VM_OK, BatchVMResult, TorchMarketPanel
from alpha_etf.scoring import ScorerConfig


SCORE_OK = 0
SCORE_VM_INVALID = 1
SCORE_NON_FINITE_RESULT = 2
SCORE_LOW_COVERAGE = 3
SCORE_CONSTANT_SIGNAL = 4
SCORE_SCORER_NO_VALID_DAYS = 5


INVALID_CODE_TO_REASON = {
    SCORE_OK: "",
    SCORE_VM_INVALID: "vm_invalid",
    SCORE_NON_FINITE_RESULT: "non_finite_result",
    SCORE_LOW_COVERAGE: "low_coverage",
    SCORE_CONSTANT_SIGNAL: "constant_signal",
    SCORE_SCORER_NO_VALID_DAYS: "scorer_no_valid_days",
}


@dataclass(frozen=True)
class BatchScoreResult:
    reward: torch.Tensor
    valid: torch.Tensor
    invalid_code: torch.Tensor
    finite_count: torch.Tensor
    coverage: torch.Tensor
    finite_std: torch.Tensor
    scorer_days: torch.Tensor
    scorer_mean_return: torch.Tensor
    scorer_hit_rate: torch.Tensor
    avg_top_k: torch.Tensor
    max_abs_signal: torch.Tensor


def _signal_quality(
    signal: torch.Tensor,
    mask: torch.Tensor,
    min_coverage: float,
    constant_std_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    usable = torch.isfinite(signal) & mask.unsqueeze(0)
    finite_count = usable.flatten(1).sum(dim=1)
    total_count = int(mask.sum().item())
    coverage = finite_count.to(signal.dtype) / float(total_count) if total_count else torch.zeros_like(finite_count, dtype=signal.dtype)

    cleaned = torch.where(usable, signal, torch.zeros_like(signal))
    denom = finite_count.clamp_min(1).to(signal.dtype)
    mean = cleaned.flatten(1).sum(dim=1) / denom
    centered = torch.where(usable, signal - mean[:, None, None], torch.zeros_like(signal))
    var = (centered * centered).flatten(1).sum(dim=1) / denom
    finite_std = torch.sqrt(torch.clamp(var, min=0.0))

    non_finite = finite_count == 0
    low_coverage = (~non_finite) & (coverage < float(min_coverage))
    constant = (~non_finite) & (~low_coverage) & (finite_std <= float(constant_std_eps))
    quality_valid = (~non_finite) & (~low_coverage) & (~constant)
    return quality_valid, non_finite, low_coverage, constant, finite_count, coverage, finite_std


def score_vm_batch(
    vm_result: BatchVMResult,
    panel: TorchMarketPanel,
    config: FormulaScoreConfig,
    scorer_config: ScorerConfig | None = None,
) -> BatchScoreResult:
    scorer_config = scorer_config or ScorerConfig(horizon=config.horizon)
    signal = vm_result.signal
    batch_size, _, time_steps = signal.shape
    dtype = signal.dtype
    device = signal.device

    reward = torch.full((batch_size,), float(config.hard_invalid_reward), dtype=dtype, device=device)
    invalid_code = torch.full((batch_size,), SCORE_VM_INVALID, dtype=torch.long, device=device)

    quality = _signal_quality(signal, panel.mask, config.min_coverage, config.constant_std_eps)
    quality_valid, non_finite, low_coverage, constant, finite_count, coverage, finite_std = quality
    vm_valid = vm_result.valid & (vm_result.invalid_code == VM_OK)

    invalid_code[vm_valid & non_finite] = SCORE_NON_FINITE_RESULT
    invalid_code[vm_valid & low_coverage] = SCORE_LOW_COVERAGE
    invalid_code[vm_valid & constant] = SCORE_CONSTANT_SIGNAL
    weak_invalid = vm_valid & (low_coverage | constant)
    reward[weak_invalid] = float(config.weak_invalid_reward)

    trainable = vm_valid & quality_valid
    scorer_days = torch.zeros(batch_size, dtype=torch.long, device=device)
    sum_returns = torch.zeros(batch_size, dtype=dtype, device=device)
    positive_days = torch.zeros(batch_size, dtype=dtype, device=device)
    sum_top_k = torch.zeros(batch_size, dtype=dtype, device=device)

    open_prices = panel.qfq("open")
    horizon = int(scorer_config.horizon)
    t_max = time_steps - horizon - 2
    for t in range(max(t_max, -1) + 1):
        available = panel.mask[:, t]
        available_count = int(available.sum().item())
        if available_count < int(scorer_config.min_universe):
            continue

        buy_t = t + 1
        sell_t = t + 1 + horizon
        base_executable = (
            available
            & panel.mask[:, buy_t]
            & panel.mask[:, sell_t]
            & torch.isfinite(open_prices[:, buy_t])
            & torch.isfinite(open_prices[:, sell_t])
            & (open_prices[:, buy_t] > 0)
        )
        if int(base_executable.sum().item()) < int(scorer_config.min_top_k):
            continue

        executable = base_executable.unsqueeze(0) & torch.isfinite(signal[:, :, t])
        eligible_count = executable.sum(dim=1)
        valid_day = trainable & (eligible_count >= int(scorer_config.min_top_k))
        if not bool(valid_day.any().item()):
            continue

        raw_k = int(math.ceil(available_count * float(scorer_config.top_fraction)))
        day_k = min(int(scorer_config.max_top_k), max(int(scorer_config.min_top_k), raw_k))
        top_k_count = torch.minimum(eligible_count, torch.full_like(eligible_count, day_k))

        ranked_signal = torch.where(executable, signal[:, :, t], torch.full_like(signal[:, :, t], -torch.inf))
        _, selected_idx = torch.topk(ranked_signal, k=day_k, dim=1)
        forward_returns_all = open_prices[:, sell_t] / open_prices[:, buy_t] - 1.0
        selected_returns = forward_returns_all[selected_idx]
        rank_pos = torch.arange(day_k, device=device).unsqueeze(0)
        selected_mask = rank_pos < top_k_count.unsqueeze(1)
        daily_mean = torch.where(selected_mask, selected_returns, torch.zeros_like(selected_returns)).sum(dim=1)
        daily_mean = daily_mean / top_k_count.clamp_min(1).to(dtype)

        sum_returns[valid_day] += daily_mean[valid_day]
        positive_days[valid_day] += (daily_mean[valid_day] > 0).to(dtype)
        sum_top_k[valid_day] += top_k_count[valid_day].to(dtype)
        scorer_days[valid_day] += 1

    has_scorer_days = scorer_days > 0
    scorer_mean_return = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    scorer_hit_rate = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    avg_top_k = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
    scorer_mean_return[has_scorer_days] = sum_returns[has_scorer_days] / scorer_days[has_scorer_days].to(dtype)
    scorer_hit_rate[has_scorer_days] = positive_days[has_scorer_days] / scorer_days[has_scorer_days].to(dtype)
    avg_top_k[has_scorer_days] = sum_top_k[has_scorer_days] / scorer_days[has_scorer_days].to(dtype)

    valid = trainable & has_scorer_days & torch.isfinite(scorer_mean_return)
    reward[valid] = scorer_mean_return[valid]
    invalid_code[trainable & ~valid] = SCORE_SCORER_NO_VALID_DAYS
    invalid_code[valid] = SCORE_OK

    abs_signal = torch.where(torch.isfinite(signal), torch.abs(signal), torch.zeros_like(signal))
    max_abs_signal = abs_signal.flatten(1).amax(dim=1)
    return BatchScoreResult(
        reward=reward,
        valid=valid,
        invalid_code=invalid_code,
        finite_count=finite_count,
        coverage=coverage,
        finite_std=finite_std,
        scorer_days=scorer_days,
        scorer_mean_return=scorer_mean_return,
        scorer_hit_rate=scorer_hit_rate,
        avg_top_k=avg_top_k,
        max_abs_signal=max_abs_signal,
    )
