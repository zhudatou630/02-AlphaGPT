"""Torch batch VM for Phase 3c GPU-first formula training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from alpha_etf.gpt.vocab import FORMULA_VOCAB, FormulaVocab
from alpha_etf.panel import MarketPanel


@dataclass(frozen=True)
class TorchMarketPanel:
    raw_values: torch.Tensor
    qfq_values: torch.Tensor
    mask: torch.Tensor
    feature_names: tuple[str, ...]

    @classmethod
    def from_market_panel(
        cls,
        panel: MarketPanel,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "TorchMarketPanel":
        return cls(
            raw_values=torch.as_tensor(panel.raw_values, dtype=dtype, device=device),
            qfq_values=torch.as_tensor(panel.qfq_values, dtype=dtype, device=device),
            mask=torch.as_tensor(panel.mask, dtype=torch.bool, device=device),
            feature_names=tuple(str(item) for item in panel.features),
        )

    def feature_index(self, feature: str) -> int:
        try:
            return self.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(f"Feature not found: {feature}") from exc

    def raw(self, feature: str) -> torch.Tensor:
        return self.raw_values[:, self.feature_index(feature), :]

    def qfq(self, feature: str) -> torch.Tensor:
        return self.qfq_values[:, self.feature_index(feature), :]


@dataclass(frozen=True)
class BatchVMResult:
    valid: torch.Tensor
    signal: torch.Tensor
    invalid_code: torch.Tensor


VM_OK = 0
VM_UNKNOWN_TOKEN = 1
VM_STACK_UNDERFLOW = 2
VM_STACK_NOT_SINGLE = 3
VM_NON_FINITE_RESULT = 4


INVALID_CODE_TO_REASON = {
    VM_OK: "",
    VM_UNKNOWN_TOKEN: "unknown_token",
    VM_STACK_UNDERFLOW: "stack_underflow",
    VM_STACK_NOT_SINGLE: "stack_not_single",
    VM_NON_FINITE_RESULT: "non_finite_result",
}


def formulas_to_tensor(formulas: list[list[int]], *, device: torch.device, pad_value: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    if not formulas:
        raise ValueError("formulas must not be empty")
    max_len = max(len(tokens) for tokens in formulas)
    if max_len < 1:
        raise ValueError("formulas must contain at least one token")
    tokens = torch.full((len(formulas), max_len), int(pad_value), dtype=torch.long, device=device)
    lengths = torch.empty(len(formulas), dtype=torch.long, device=device)
    for row, formula in enumerate(formulas):
        if not formula:
            raise ValueError("empty formula is not supported")
        values = torch.as_tensor(formula, dtype=torch.long, device=device)
        tokens[row, : values.numel()] = values
        lengths[row] = int(values.numel())
    return tokens, lengths


def finite_or_nan(values: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(values), values, torch.full_like(values, float("nan")))


def delay(values: torch.Tensor, periods: int) -> torch.Tensor:
    if periods <= 0:
        return values.clone()
    out = torch.full_like(values, float("nan"))
    if periods < values.shape[-1]:
        out[..., periods:] = values[..., :-periods]
    return out


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return finite_or_nan(numerator / denominator)


def ret(values: torch.Tensor, periods: int) -> torch.Tensor:
    return safe_divide(values, delay(values, periods)) - 1.0


def decay(values: torch.Tensor) -> torch.Tensor:
    return finite_or_nan((values + 0.8 * delay(values, 1) + 0.6 * delay(values, 2)) / 2.4)


def _rolling_sum(values: torch.Tensor, window: int) -> torch.Tensor:
    padded = F.pad(values, (1, 0), value=0.0)
    cumsum = padded.cumsum(dim=-1)
    sums = cumsum[..., window:] - cumsum[..., :-window]
    out = torch.full_like(values, float("nan"))
    out[..., window - 1 :] = sums
    return out


def rolling_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    finite = torch.isfinite(values)
    cleaned = torch.where(finite, values, torch.zeros_like(values))
    counts = _rolling_sum(finite.to(values.dtype), window)
    sums = _rolling_sum(cleaned, window)
    mean = sums / float(window)
    return torch.where(counts == float(window), mean, torch.full_like(values, float("nan")))


def rolling_std(values: torch.Tensor, window: int) -> torch.Tensor:
    finite = torch.isfinite(values)
    cleaned = torch.where(finite, values, torch.zeros_like(values))
    counts = _rolling_sum(finite.to(values.dtype), window)
    sums = _rolling_sum(cleaned, window)
    sumsq = _rolling_sum(cleaned * cleaned, window)
    mean = sums / float(window)
    var = torch.clamp(sumsq / float(window) - mean * mean, min=0.0)
    std = torch.sqrt(var)
    return torch.where(counts == float(window), std, torch.full_like(values, float("nan")))


def apply_operator(name: str, args: list[torch.Tensor]) -> torch.Tensor:
    if name == "ADD":
        return finite_or_nan(args[0] + args[1])
    if name == "SUB":
        return finite_or_nan(args[0] - args[1])
    if name == "MUL":
        return finite_or_nan(args[0] * args[1])
    if name == "DIV":
        return safe_divide(args[0], args[1])
    if name == "NEG":
        return finite_or_nan(-args[0])
    if name == "ABS":
        return finite_or_nan(torch.abs(args[0]))
    if name == "SIGN":
        return finite_or_nan(torch.sign(args[0]))
    if name == "DELAY1":
        return delay(args[0], 1)
    if name == "DELAY5":
        return delay(args[0], 5)
    if name == "DELAY10":
        return delay(args[0], 10)
    if name == "MA5":
        return rolling_mean(args[0], 5)
    if name == "MA10":
        return rolling_mean(args[0], 10)
    if name == "MA20":
        return rolling_mean(args[0], 20)
    if name == "STD10":
        return rolling_std(args[0], 10)
    if name == "RET5":
        return ret(args[0], 5)
    if name == "RET10":
        return ret(args[0], 10)
    if name == "DECAY":
        return decay(args[0])
    raise KeyError(f"Unsupported operator: {name}")


class BatchTorchVM:
    def __init__(self, vocab: FormulaVocab = FORMULA_VOCAB):
        self.vocab = vocab

    def execute(self, token_ids: torch.Tensor, lengths: torch.Tensor, panel: TorchMarketPanel) -> BatchVMResult:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape [batch, max_len], got {tuple(token_ids.shape)}")
        batch_size, max_len = token_ids.shape
        if lengths.shape != (batch_size,):
            raise ValueError(f"lengths must have shape [{batch_size}], got {tuple(lengths.shape)}")

        assets, time_steps = panel.mask.shape
        stack = torch.empty(
            (batch_size, max_len, assets, time_steps),
            dtype=panel.qfq_values.dtype,
            device=token_ids.device,
        )
        depths = torch.zeros(batch_size, dtype=torch.long, device=token_ids.device)
        invalid_code = torch.zeros(batch_size, dtype=torch.long, device=token_ids.device)

        for pos in range(max_len):
            active = (pos < lengths) & (invalid_code == VM_OK)
            if not bool(active.any().item()):
                continue
            current = token_ids[:, pos]
            for formula_token_id, token in enumerate(self.vocab.tokens):
                rows = active & (current == formula_token_id)
                if not bool(rows.any().item()):
                    continue
                row_idx = torch.where(rows)[0]
                row_depths = depths[row_idx]
                if token.kind == "feature":
                    source = panel.qfq(token.name) if token.source == "qfq" else panel.raw(token.name)
                    stack[row_idx, row_depths] = source.unsqueeze(0)
                    depths[row_idx] = row_depths + 1
                    continue
                if token.kind == "constant":
                    value = 0.0 if token.value is None else float(token.value)
                    stack[row_idx, row_depths] = value
                    depths[row_idx] = row_depths + 1
                    continue
                if token.kind != "operator":
                    invalid_code[row_idx] = VM_UNKNOWN_TOKEN
                    continue
                underflow = row_depths < token.arity
                if bool(underflow.any().item()):
                    invalid_code[row_idx[underflow]] = VM_STACK_UNDERFLOW
                ok_idx = row_idx[~underflow]
                if ok_idx.numel() == 0:
                    continue
                ok_depths = depths[ok_idx]
                if token.arity == 1:
                    arg = stack[ok_idx, ok_depths - 1]
                    stack[ok_idx, ok_depths - 1] = apply_operator(token.name, [arg])
                elif token.arity == 2:
                    left = stack[ok_idx, ok_depths - 2]
                    right = stack[ok_idx, ok_depths - 1]
                    stack[ok_idx, ok_depths - 2] = apply_operator(token.name, [left, right])
                    depths[ok_idx] = ok_depths - 1
                else:
                    invalid_code[ok_idx] = VM_UNKNOWN_TOKEN

        not_single = (depths != 1) & (invalid_code == VM_OK)
        invalid_code[not_single] = VM_STACK_NOT_SINGLE
        signal = stack[:, 0]
        non_finite = (~torch.isfinite(signal).flatten(1).any(dim=1)) & (invalid_code == VM_OK)
        invalid_code[non_finite] = VM_NON_FINITE_RESULT
        valid = invalid_code == VM_OK
        signal = torch.where(valid[:, None, None], signal, torch.full_like(signal, float("nan")))
        return BatchVMResult(valid=valid, signal=finite_or_nan(signal), invalid_code=invalid_code)
