"""NumPy reference VM for compiled V3A formulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_etf.research_v3a.factors import FACTOR_NAMES, rolling_mean_numpy
from alpha_etf.research_v3a.language import (
    ABS_CODE,
    ADD_CODE,
    CONST_0_CODE,
    CONST_1_CODE,
    MEAN_CODE_BY_WINDOW,
    MUL_CODE,
    NEG_CODE,
    REF_CODE_BY_WINDOW,
    SIGN_CODE,
    SUB_CODE,
    CompiledFormula,
    compile_formula,
)


@dataclass(frozen=True)
class VMResult:
    valid: bool
    signal: np.ndarray | None = None
    invalid_reason: str = ""


def _delay(values: np.ndarray, periods: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    if periods < values.shape[-1]:
        out[:, periods:] = values[:, :-periods]
    return out


class StackVM:
    def execute_compiled(
        self, compiled: CompiledFormula, factor_values: np.ndarray, mask: np.ndarray
    ) -> VMResult:
        if factor_values.shape != (len(FACTOR_NAMES), mask.shape[0], mask.shape[1]):
            raise ValueError(f"V3A factor cache differs from mask: {factor_values.shape}")
        stack: list[np.ndarray] = []
        ref_by_code = {code: window for window, code in REF_CODE_BY_WINDOW.items()}
        mean_by_code = {code: window for window, code in MEAN_CODE_BY_WINDOW.items()}
        try:
            for code in compiled.instructions:
                if 0 <= code < len(FACTOR_NAMES):
                    stack.append(factor_values[code].astype(np.float64, copy=True))
                elif code == CONST_0_CODE or code == CONST_1_CODE:
                    value = 0.0 if code == CONST_0_CODE else 1.0
                    stack.append(np.where(mask, value, np.nan).astype(np.float64))
                elif code in {NEG_CODE, ABS_CODE, SIGN_CODE}:
                    if not stack:
                        return VMResult(False, invalid_reason="stack_underflow")
                    arg = stack.pop()
                    if code == NEG_CODE:
                        result = -arg
                    elif code == ABS_CODE:
                        result = np.abs(arg)
                    else:
                        result = np.sign(arg)
                    stack.append(np.where(mask, result, np.nan))
                elif code in {ADD_CODE, SUB_CODE, MUL_CODE}:
                    if len(stack) < 2:
                        return VMResult(False, invalid_reason="stack_underflow")
                    right = stack.pop()
                    left = stack.pop()
                    if code == ADD_CODE:
                        result = left + right
                    elif code == SUB_CODE:
                        result = left - right
                    else:
                        result = left * right
                    result[~np.isfinite(result)] = np.nan
                    stack.append(np.where(mask, result, np.nan))
                elif code in ref_by_code:
                    if not stack:
                        return VMResult(False, invalid_reason="stack_underflow")
                    stack.append(np.where(mask, _delay(stack.pop(), ref_by_code[code]), np.nan))
                elif code in mean_by_code:
                    if not stack:
                        return VMResult(False, invalid_reason="stack_underflow")
                    stack.append(
                        np.where(mask, rolling_mean_numpy(stack.pop(), mean_by_code[code]), np.nan)
                    )
                else:
                    return VMResult(False, invalid_reason="unknown_instruction")
            if len(stack) != 1:
                return VMResult(False, invalid_reason="stack_not_single")
            signal = stack[0]
            if not np.isfinite(signal).any():
                return VMResult(False, signal=signal, invalid_reason="non_finite_result")
            return VMResult(True, signal=signal)
        except (FloatingPointError, RuntimeError, ValueError):
            return VMResult(False, invalid_reason="operator_error")

    def execute(
        self, token_ids: list[int] | tuple[int, ...], factor_values: np.ndarray, mask: np.ndarray
    ) -> VMResult:
        try:
            compiled = compile_formula(token_ids)
        except (KeyError, ValueError):
            return VMResult(False, invalid_reason="invalid_formula")
        return self.execute_compiled(compiled, factor_values, mask)