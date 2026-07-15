"""Batch Torch aggregation for precomputed V3A causal targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alpha_etf.research_v3a.scoring import ForwardTargets, ScorerConfig


SCORE_OK = 0
SCORE_VM_INVALID = 1
SCORE_INSUFFICIENT_DAILY_SIGNAL = 2


@dataclass(frozen=True)
class TorchForwardTargets:
    decision_indices: torch.Tensor
    available: torch.Tensor
    top_k: torch.Tensor
    forward_returns: torch.Tensor
    baseline_returns: torch.Tensor

    @classmethod
    def from_numpy(
        cls, targets: ForwardTargets, *, device: torch.device, dtype: torch.dtype
    ) -> "TorchForwardTargets":
        return cls(
            decision_indices=torch.as_tensor(
                targets.decision_indices, dtype=torch.long, device=device
            ),
            available=torch.as_tensor(targets.available, dtype=torch.bool, device=device),
            top_k=torch.as_tensor(targets.top_k, dtype=torch.long, device=device),
            forward_returns=torch.as_tensor(
                targets.forward_returns, dtype=dtype, device=device
            ),
            baseline_returns=torch.as_tensor(
                targets.baseline_returns, dtype=dtype, device=device
            ),
        )


@dataclass(frozen=True)
class BatchScoreResult:
    reward: torch.Tensor
    valid: torch.Tensor
    invalid_code: torch.Tensor
    daily_absolute_return: torch.Tensor
    daily_excess_return: torch.Tensor
    selected_indices: torch.Tensor


def signal_quality_batch(
    signals: torch.Tensor,
    targets: TorchForwardTargets,
    *,
    min_coverage: float,
    constant_std_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure finite coverage and cross-panel variation for each formula."""

    decision = signals.index_select(2, targets.decision_indices).permute(0, 2, 1)
    usable = targets.available.unsqueeze(0) & torch.isfinite(decision)
    finite_count = usable.sum(dim=(1, 2))
    total = targets.available.sum().to(signals.dtype)
    coverage = finite_count.to(signals.dtype) / total
    cleaned = torch.where(usable, decision, torch.zeros_like(decision))
    denominator = finite_count.clamp_min(1).to(signals.dtype)
    mean = cleaned.sum(dim=(1, 2)) / denominator
    variance = (
        torch.where(
            usable,
            (decision - mean[:, None, None]) ** 2,
            torch.zeros_like(decision),
        ).sum(dim=(1, 2))
        / denominator
    )
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    coverage_valid = coverage >= float(min_coverage)
    variation_valid = torch.isfinite(std) & (std > float(constant_std_eps))
    return coverage_valid & variation_valid, coverage, std, variation_valid


def signal_quality_batch_chunked(
    signals: torch.Tensor,
    targets: TorchForwardTargets,
    *,
    min_coverage: float,
    constant_std_eps: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size < 1:
        raise ValueError("Torch quality chunk size must be positive")
    parts = [
        signal_quality_batch(
            signals[start : start + chunk_size],
            targets,
            min_coverage=min_coverage,
            constant_std_eps=constant_std_eps,
        )
        for start in range(0, signals.shape[0], chunk_size)
    ]
    return tuple(torch.cat(values, dim=0) for values in zip(*parts))


def score_signal_batch(
    signals: torch.Tensor,
    vm_valid: torch.Tensor,
    targets: TorchForwardTargets,
    config: ScorerConfig,
) -> BatchScoreResult:
    if signals.ndim != 3:
        raise ValueError(f"signals must be [batch,asset,date], got {tuple(signals.shape)}")
    if signals.dtype != torch.float32 or config.ranking_dtype != "float32":
        raise ValueError("V3A Torch scorer requires float32 ranking signals")
    batch, assets, _ = signals.shape
    if vm_valid.shape != (batch,):
        raise ValueError("vm_valid differs from signals batch")
    days = targets.decision_indices.numel()
    if targets.available.shape != (days, assets):
        raise ValueError("Torch target availability differs from signal assets")

    decision_signal = signals.index_select(2, targets.decision_indices).permute(0, 2, 1)
    eligible = targets.available.unsqueeze(0) & torch.isfinite(decision_signal)
    finite_count = eligible.sum(dim=-1)
    enough_each_day = finite_count >= targets.top_k.unsqueeze(0)
    enough_all_days = enough_each_day.all(dim=1)
    valid = vm_valid & enough_all_days

    ranked_signal = torch.where(
        eligible, decision_signal, torch.full_like(decision_signal, -torch.inf)
    )
    max_k = int(targets.top_k.max().item())
    ordered = torch.argsort(ranked_signal, dim=-1, descending=True, stable=True)
    selected = ordered[..., :max_k]
    returns = targets.forward_returns.unsqueeze(0).expand(batch, -1, -1)
    selected_returns = torch.gather(returns, 2, selected)
    positions = torch.arange(max_k, device=signals.device).view(1, 1, -1)
    selected_mask = positions < targets.top_k.view(1, -1, 1)
    daily_absolute = torch.where(
        selected_mask, selected_returns, torch.zeros_like(selected_returns)
    ).sum(dim=-1) / targets.top_k.to(signals.dtype).unsqueeze(0)
    daily_excess = daily_absolute - targets.baseline_returns.unsqueeze(0)
    reward = daily_excess.mean(dim=1)
    reward = torch.where(valid, reward, torch.full_like(reward, config.hard_invalid_reward))
    invalid_code = torch.full((batch,), SCORE_VM_INVALID, dtype=torch.long, device=signals.device)
    invalid_code[vm_valid & ~enough_all_days] = SCORE_INSUFFICIENT_DAILY_SIGNAL
    invalid_code[valid] = SCORE_OK
    return BatchScoreResult(
        reward=reward,
        valid=valid,
        invalid_code=invalid_code,
        daily_absolute_return=daily_absolute,
        daily_excess_return=daily_excess,
        selected_indices=selected,
    )


def score_signal_batch_chunked(
    signals: torch.Tensor,
    vm_valid: torch.Tensor,
    targets: TorchForwardTargets,
    config: ScorerConfig,
    *,
    chunk_size: int,
) -> BatchScoreResult:
    if chunk_size < 1:
        raise ValueError("Torch scorer chunk size must be positive")
    parts = [
        score_signal_batch(
            signals[start : start + chunk_size],
            vm_valid[start : start + chunk_size],
            targets,
            config,
        )
        for start in range(0, signals.shape[0], chunk_size)
    ]
    return BatchScoreResult(
        reward=torch.cat([part.reward for part in parts], dim=0),
        valid=torch.cat([part.valid for part in parts], dim=0),
        invalid_code=torch.cat([part.invalid_code for part in parts], dim=0),
        daily_absolute_return=torch.cat(
            [part.daily_absolute_return for part in parts], dim=0
        ),
        daily_excess_return=torch.cat(
            [part.daily_excess_return for part in parts], dim=0
        ),
        selected_indices=torch.cat([part.selected_indices for part in parts], dim=0),
    )