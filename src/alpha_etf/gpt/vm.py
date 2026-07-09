"""StackVM for Phase 3a token formulas."""

from __future__ import annotations

from dataclasses import dataclass

import traceback

import numpy as np

from alpha_etf.gpt.ops import OPS, finite_or_nan
from alpha_etf.gpt.vocab import FORMULA_VOCAB, FormulaVocab
from alpha_etf.panel import MarketPanel


@dataclass(frozen=True)
class VMResult:
    valid: bool
    signal: np.ndarray | None = None
    invalid_reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    invalid_reason: str = ""
    finite_count: int = 0
    coverage: float = 0.0
    finite_std: float = np.nan


class StackVM:
    def __init__(self, vocab: FormulaVocab = FORMULA_VOCAB):
        self.vocab = vocab

    def execute(self, token_ids: list[int] | tuple[int, ...], panel: MarketPanel) -> VMResult:
        stack: list[np.ndarray] = []
        target_shape = panel.mask.shape

        try:
            for raw_token_id in token_ids:
                token_id = int(raw_token_id)
                try:
                    token = self.vocab.id_to_token(token_id)
                except KeyError:
                    return VMResult(valid=False, invalid_reason="unknown_token")

                if token.kind == "feature":
                    if token.source == "qfq":
                        stack.append(panel.qfq(token.name).astype(float).copy())
                    elif token.source == "raw":
                        stack.append(panel.raw(token.name).astype(float).copy())
                    else:
                        return VMResult(valid=False, invalid_reason="unknown_feature_source")
                    continue

                if token.kind == "constant":
                    if token.value is None:
                        return VMResult(valid=False, invalid_reason="constant_without_value")
                    stack.append(np.full(target_shape, float(token.value), dtype=float))
                    continue

                if token.kind != "operator":
                    return VMResult(valid=False, invalid_reason="unknown_token_kind")

                op = OPS.get(token.name)
                if op is None:
                    return VMResult(valid=False, invalid_reason="operator_not_implemented")
                if len(stack) < op.arity:
                    return VMResult(valid=False, invalid_reason="stack_underflow")

                args = stack[-op.arity:]
                del stack[-op.arity:]
                result = finite_or_nan(op.func(*args))
                if result.shape != target_shape:
                    return VMResult(valid=False, invalid_reason="shape_mismatch")
                stack.append(result)

            if len(stack) != 1:
                return VMResult(valid=False, invalid_reason="stack_not_single")

            signal = finite_or_nan(stack[0])
            if not np.isfinite(signal).any():
                return VMResult(valid=False, invalid_reason="non_finite_result")
            return VMResult(valid=True, signal=signal)
        except Exception as exc:  # pragma: no cover - defensive guard for generated formulas.
            return VMResult(valid=False, invalid_reason="operator_error", error=f"{exc}\n{traceback.format_exc(limit=3)}")


def check_signal_quality(
    signal: np.ndarray,
    mask: np.ndarray,
    min_coverage: float = 0.20,
    constant_std_eps: float = 1e-12,
) -> QualityResult:
    usable = mask & np.isfinite(signal)
    finite_count = int(usable.sum())
    total_count = int(mask.sum())
    coverage = finite_count / total_count if total_count else 0.0

    if finite_count == 0:
        return QualityResult(False, "non_finite_result", finite_count, coverage, np.nan)
    if coverage < min_coverage:
        return QualityResult(False, "low_coverage", finite_count, coverage, np.nan)

    values = signal[usable]
    finite_std = float(np.nanstd(values))
    if not np.isfinite(finite_std) or finite_std <= constant_std_eps:
        return QualityResult(False, "constant_signal", finite_count, coverage, finite_std)

    return QualityResult(True, finite_count=finite_count, coverage=coverage, finite_std=finite_std)
