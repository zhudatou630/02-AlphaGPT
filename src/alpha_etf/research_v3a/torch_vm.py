"""Batch Torch VM for compiled V3A formulas."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from alpha_etf.research_v3a.factors import FACTOR_NAMES, rolling_mean_torch
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
)


VM_OK = 0
VM_STACK_UNDERFLOW = 1
VM_STACK_NOT_SINGLE = 2
VM_NON_FINITE_RESULT = 3
VM_UNKNOWN_INSTRUCTION = 4


@dataclass(frozen=True)
class BatchVMResult:
    valid: torch.Tensor
    signal: torch.Tensor
    invalid_code: torch.Tensor


@dataclass(frozen=True)
class ExecutionPlan:
    batch_size: int
    max_stack_depth: int
    chunk_size: int
    estimated_working_bytes: int
    output_bytes: int
    estimated_peak_bytes: int


def compiled_to_tensor(
    formulas: list[CompiledFormula], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not formulas:
        raise ValueError("compiled formulas must not be empty")
    max_length = max(len(formula.instructions) for formula in formulas)
    codes = torch.full((len(formulas), max_length), -1, dtype=torch.long, device=device)
    lengths = torch.empty(len(formulas), dtype=torch.long, device=device)
    for row, formula in enumerate(formulas):
        values = torch.as_tensor(formula.instructions, dtype=torch.long, device=device)
        codes[row, : len(formula.instructions)] = values
        lengths[row] = len(formula.instructions)
    return codes, lengths


def _delay(values: torch.Tensor, periods: int) -> torch.Tensor:
    out = torch.full_like(values, float("nan"))
    if periods < values.shape[-1]:
        out[..., periods:] = values[..., :-periods]
    return out


class BatchTorchVM:
    def __init__(
        self,
        *,
        max_working_bytes: int = 512 * 1024**2,
        max_output_bytes: int = 2 * 1024**3,
        max_total_bytes: int = 3 * 1024**3,
    ):
        if max_working_bytes < 1 or max_output_bytes < 1 or max_total_bytes < 1:
            raise ValueError("V3A Torch VM memory budgets must be positive")
        self.max_working_bytes = int(max_working_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self.max_total_bytes = int(max_total_bytes)

    @staticmethod
    def _max_stack_depth(codes: torch.Tensor, lengths: torch.Tensor) -> int:
        max_depth = 0
        code_rows = codes.detach().cpu().tolist()
        length_rows = lengths.detach().cpu().tolist()
        for row, length in zip(code_rows, length_rows, strict=True):
            depth = 0
            row_max = 0
            for code in row[: int(length)]:
                if 0 <= code < len(FACTOR_NAMES) or code in {CONST_0_CODE, CONST_1_CODE}:
                    depth += 1
                elif code in {ADD_CODE, SUB_CODE, MUL_CODE}:
                    depth -= 1
                row_max = max(row_max, depth)
            if depth != 1 or row_max < 1:
                raise ValueError("Compiled V3A formula has an invalid stack profile")
            max_depth = max(max_depth, row_max)
        return max_depth

    def execution_plan(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        factor_values: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_stack_depth: int | None = None,
    ) -> ExecutionPlan:
        if codes.ndim != 2 or lengths.shape != (codes.shape[0],):
            raise ValueError("Compiled V3A tensor shapes differ")
        if factor_values.shape != (len(FACTOR_NAMES), mask.shape[0], mask.shape[1]):
            raise ValueError(f"factor cache differs from mask: {tuple(factor_values.shape)}")
        batch_size = int(codes.shape[0])
        max_depth = (
            self._max_stack_depth(codes, lengths)
            if max_stack_depth is None
            else int(max_stack_depth)
        )
        if max_depth < 1:
            raise ValueError("Compiled V3A formulas require a positive stack depth")
        assets, dates = mask.shape
        element_size = factor_values.element_size()
        output_bytes = batch_size * assets * dates * element_size
        if output_bytes > self.max_output_bytes:
            raise MemoryError(
                "V3A Torch VM output exceeds its memory gate; slice the research date range or "
                f"reduce the formula batch ({output_bytes} > {self.max_output_bytes} bytes)"
            )
        # The extra three panels cover left/right/result temporaries of a binary operation.
        per_formula_working = (max_depth + 3) * assets * dates * element_size
        available_working = min(self.max_working_bytes, self.max_total_bytes - output_bytes)
        if available_working < per_formula_working:
            raise MemoryError(
                "One V3A formula chunk exceeds the total memory gate after reserving output"
            )
        chunk_size = max(1, min(batch_size, available_working // per_formula_working))
        estimated = chunk_size * per_formula_working
        estimated_peak = output_bytes + estimated
        return ExecutionPlan(
            batch_size,
            max_depth,
            chunk_size,
            estimated,
            output_bytes,
            estimated_peak,
        )

    def execute(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        factor_values: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_stack_depth: int | None = None,
    ) -> BatchVMResult:
        if not (codes.device == lengths.device == factor_values.device == mask.device):
            raise ValueError("V3A Torch VM inputs must be on the same device")
        plan = self.execution_plan(
            codes,
            lengths,
            factor_values,
            mask,
            max_stack_depth=max_stack_depth,
        )
        if plan.chunk_size >= plan.batch_size:
            return self._execute_chunk(
                codes, lengths, factor_values, mask, max_stack_depth=plan.max_stack_depth
            )
        assets, dates = mask.shape
        valid_output = torch.empty(plan.batch_size, dtype=torch.bool, device=codes.device)
        invalid_output = torch.empty(plan.batch_size, dtype=torch.long, device=codes.device)
        signal_output = torch.empty(
            (plan.batch_size, assets, dates),
            dtype=factor_values.dtype,
            device=factor_values.device,
        )
        for start in range(0, plan.batch_size, plan.chunk_size):
            stop = min(start + plan.chunk_size, plan.batch_size)
            part = self._execute_chunk(
                codes[start:stop],
                lengths[start:stop],
                factor_values,
                mask,
                max_stack_depth=plan.max_stack_depth,
            )
            valid_output[start:stop] = part.valid
            invalid_output[start:stop] = part.invalid_code
            signal_output[start:stop] = part.signal
        return BatchVMResult(
            valid=valid_output,
            signal=signal_output,
            invalid_code=invalid_output,
        )

    def _execute_chunk(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        factor_values: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_stack_depth: int,
    ) -> BatchVMResult:
        batch_size, max_length = codes.shape

        assets, dates = mask.shape
        stack = torch.empty(
            (batch_size, max_stack_depth, assets, dates),
            dtype=factor_values.dtype,
            device=factor_values.device,
        )
        depths = torch.zeros(batch_size, dtype=torch.long, device=codes.device)
        invalid = torch.zeros(batch_size, dtype=torch.long, device=codes.device)
        ref_by_code = {code: window for window, code in REF_CODE_BY_WINDOW.items()}
        mean_by_code = {code: window for window, code in MEAN_CODE_BY_WINDOW.items()}
        all_codes = (
            list(range(len(FACTOR_NAMES)))
            + [CONST_0_CODE, CONST_1_CODE, ADD_CODE, SUB_CODE, MUL_CODE, NEG_CODE, ABS_CODE, SIGN_CODE]
            + list(ref_by_code)
            + list(mean_by_code)
        )

        for position in range(max_length):
            active = (position < lengths) & (invalid == VM_OK)
            if not bool(active.any().item()):
                continue
            current = codes[:, position]
            known = torch.zeros(batch_size, dtype=torch.bool, device=codes.device)
            for code in all_codes:
                rows = active & (current == code)
                if not bool(rows.any().item()):
                    continue
                known |= rows
                row_idx = torch.where(rows)[0]
                row_depths = depths[row_idx]
                if 0 <= code < len(FACTOR_NAMES):
                    stack[row_idx, row_depths] = factor_values[code].unsqueeze(0)
                    depths[row_idx] += 1
                    continue
                if code in {CONST_0_CODE, CONST_1_CODE}:
                    value = 0.0 if code == CONST_0_CODE else 1.0
                    source = torch.where(mask, torch.full_like(factor_values[0], value), torch.nan)
                    stack[row_idx, row_depths] = source.unsqueeze(0)
                    depths[row_idx] += 1
                    continue

                arity = 2 if code in {ADD_CODE, SUB_CODE, MUL_CODE} else 1
                underflow = row_depths < arity
                if bool(underflow.any().item()):
                    invalid[row_idx[underflow]] = VM_STACK_UNDERFLOW
                ok_idx = row_idx[~underflow]
                if ok_idx.numel() == 0:
                    continue
                ok_depths = depths[ok_idx]
                if arity == 2:
                    left = stack[ok_idx, ok_depths - 2]
                    right = stack[ok_idx, ok_depths - 1]
                    if code == ADD_CODE:
                        result = left + right
                    elif code == SUB_CODE:
                        result = left - right
                    else:
                        result = left * right
                    result = torch.where(mask, result, torch.full_like(result, float("nan")))
                    stack[ok_idx, ok_depths - 2] = result
                    depths[ok_idx] -= 1
                else:
                    arg = stack[ok_idx, ok_depths - 1]
                    if code == NEG_CODE:
                        result = -arg
                    elif code == ABS_CODE:
                        result = torch.abs(arg)
                    elif code == SIGN_CODE:
                        result = torch.where(
                            torch.isfinite(arg),
                            torch.sign(arg),
                            torch.full_like(arg, float("nan")),
                        )
                    elif code in ref_by_code:
                        result = _delay(arg, ref_by_code[code])
                    elif code in mean_by_code:
                        result = rolling_mean_torch(arg, mean_by_code[code])
                    else:
                        invalid[ok_idx] = VM_UNKNOWN_INSTRUCTION
                        continue
                    stack[ok_idx, ok_depths - 1] = torch.where(
                        mask, result, torch.full_like(result, float("nan"))
                    )
            unknown = active & ~known
            invalid[unknown] = VM_UNKNOWN_INSTRUCTION

        invalid[(depths != 1) & (invalid == VM_OK)] = VM_STACK_NOT_SINGLE
        signal = stack[:, 0]
        non_finite = (~torch.isfinite(signal).flatten(1).any(dim=1)) & (invalid == VM_OK)
        invalid[non_finite] = VM_NON_FINITE_RESULT
        valid = invalid == VM_OK
        signal = torch.where(valid[:, None, None], signal, torch.full_like(signal, float("nan")))
        return BatchVMResult(valid=valid, signal=signal, invalid_code=invalid)