#!/usr/bin/env python3
"""Create a non-destructive, human-reviewable formula curation preview.

The preview reads the immutable attempt ledger, applies only display-layer
simplifications, and emits several reward-tolerance ranking scenarios. It does
not rewrite the canonical ledger or the existing top-N artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


CURATION_PREVIEW_SCHEMA_VERSION = "etf-v3a-formula-curation-preview-v1"
CURATION_RULE_VERSION = "etf-v3a-display-simplifier-v1"
CURATED_REWARD_TOLERANCE = 5e-6
DEFAULT_TOLERANCES = (0.0, 1e-6, CURATED_REWARD_TOLERANCE, 1e-5)
WINDOW_PREFIX = "WIN_"
FIXED_FACTORS = {
    "DAYRET",
    "GAP",
    "INTRADAY",
    "RANGE",
    "CLV",
}
PARAMETER_FACTORS = {
    "ROC": 1,
    "PRICE_MA": 1,
    "VOL": 1,
    "TS_RANK": 1,
    "RSV": 1,
    "MA_RATIO": 2,
}
UNARY_OPERATORS = {"NEG", "ABS", "SIGN"}
BINARY_OPERATORS = {"ADD", "SUB", "MUL"}
ROLLING_OPERATORS = {"REF", "MEAN"}


@dataclass(frozen=True)
class _Window:
    value: int


@dataclass(frozen=True)
class _Expression:
    kind: str
    name: str
    params: tuple[int | float, ...] = ()
    children: tuple["_Expression", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": list(self.params),
            "children": [child.to_dict() for child in self.children],
        }

    def text(self) -> str:
        if self.kind == "factor":
            if not self.params:
                return self.name
            return f"{self.name}({','.join(str(item) for item in self.params)})"
        if self.kind == "constant":
            return self.name
        args = [child.text() for child in self.children]
        args.extend(str(item) for item in self.params)
        return f"{self.name}({','.join(args)})"

    def token_len(self) -> int:
        if self.kind in {"factor", "constant"}:
            return 1 + (len(self.params) if self.kind == "factor" else 0)
        return 1 + len(self.params) + sum(child.token_len() for child in self.children)


@dataclass(frozen=True)
class _FormulaRow:
    formula_hash: str
    token_names: tuple[str, ...]
    reward: float
    best_attempt_index: int
    first_attempt_index: int
    attempt_count: int
    raw_expression: _Expression
    simplified_expression: _Expression
    simplification_rules: tuple[str, ...]

    @property
    def raw_token_len(self) -> int:
        return len(self.token_names)

    @property
    def simplified_token_len(self) -> int:
        return self.simplified_expression.token_len()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <run-dir>/formula_library/curation_preview.",
    )
    parser.add_argument(
        "--tolerances",
        default=",".join(str(value) for value in DEFAULT_TOLERANCES),
        help="Comma-separated reward tolerances for preview scenarios.",
    )
    return parser.parse_args()


def _parse_tolerances(value: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for item in value.split(","):
        tolerance = float(item.strip())
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Reward tolerances must be finite and non-negative")
        if tolerance not in parsed:
            parsed.append(tolerance)
    if not parsed:
        raise ValueError("At least one reward tolerance is required")
    return tuple(parsed)


def _window_value(name: str) -> int:
    if not name.startswith(WINDOW_PREFIX):
        raise ValueError(f"Expected a window token, got {name}")
    return int(name.removeprefix(WINDOW_PREFIX))


def _parse_expression(token_names: Iterable[str]) -> _Expression:
    tokens = tuple(str(name) for name in token_names)
    stack: list[_Expression | _Window] = []
    index = 0
    while index < len(tokens):
        name = tokens[index]
        if name in FIXED_FACTORS:
            stack.append(_Expression("factor", name))
        elif name in PARAMETER_FACTORS:
            required = PARAMETER_FACTORS[name]
            end = index + required + 1
            if end > len(tokens) or any(
                not tokens[offset].startswith(WINDOW_PREFIX)
                for offset in range(index + 1, end)
            ):
                raise ValueError(f"Malformed parameters for {name}")
            stack.append(
                _Expression(
                    "factor",
                    name,
                    tuple(_window_value(tokens[offset]) for offset in range(index + 1, end)),
                )
            )
            index += required
        elif name in {"CONST_0", "CONST_1"}:
            stack.append(_Expression("constant", name))
        elif name.startswith(WINDOW_PREFIX):
            stack.append(_Window(_window_value(name)))
        elif name in UNARY_OPERATORS:
            if not stack or not isinstance(stack[-1], _Expression):
                raise ValueError(f"Unary operator {name} has no expression operand")
            stack.append(_Expression("operator", name, children=(stack.pop(),)))
        elif name in BINARY_OPERATORS:
            if len(stack) < 2:
                raise ValueError(f"Binary operator {name} lacks operands")
            right = stack.pop()
            left = stack.pop()
            if not isinstance(left, _Expression) or not isinstance(right, _Expression):
                raise ValueError(f"Binary operator {name} has a window operand")
            stack.append(_Expression("operator", name, children=(left, right)))
        elif name in ROLLING_OPERATORS:
            if len(stack) < 2:
                raise ValueError(f"Rolling operator {name} lacks operands")
            window = stack.pop()
            value = stack.pop()
            if not isinstance(window, _Window) or not isinstance(value, _Expression):
                raise ValueError(f"Rolling operator {name} has malformed operands")
            stack.append(_Expression("operator", name, (window.value,), (value,)))
        else:
            raise ValueError(f"Unknown V3A token {name}")
        index += 1

    if len(stack) != 1 or not isinstance(stack[0], _Expression):
        raise ValueError("Formula did not compile to one expression")
    return stack[0]


def _constant_value(expression: _Expression) -> float | None:
    if expression.kind == "constant":
        if expression.name == "CONST_0":
            return 0.0
        if expression.name == "CONST_1":
            return 1.0
        return None
    if expression.kind != "operator" or expression.name not in UNARY_OPERATORS:
        return None
    if len(expression.children) != 1:
        return None
    child_value = _constant_value(expression.children[0])
    if child_value is None:
        return None
    if expression.name == "NEG":
        return -child_value
    if expression.name == "ABS":
        return abs(child_value)
    return float(0.0 if child_value == 0.0 else 1.0 if child_value > 0.0 else -1.0)


def _constant(value: float) -> _Expression | None:
    if value == 0.0:
        return _Expression("constant", "CONST_0")
    if value == 1.0:
        return _Expression("constant", "CONST_1")
    if value == -1.0:
        return _Expression(
            "operator",
            "NEG",
            children=(_Expression("constant", "CONST_1"),),
        )
    return None


def _expression_key(expression: _Expression) -> str:
    return json.dumps(expression.to_dict(), sort_keys=True, separators=(",", ":"))


def _simplify_expression(
    expression: _Expression,
) -> tuple[_Expression, tuple[str, ...]]:
    if not expression.children:
        return expression, ()

    simplified_children: list[_Expression] = []
    rules: list[str] = []
    for child in expression.children:
        simplified_child, child_rules = _simplify_expression(child)
        simplified_children.append(simplified_child)
        rules.extend(child_rules)

    name = expression.name
    children = tuple(simplified_children)

    if name in UNARY_OPERATORS:
        child = children[0]
        value = _constant_value(child)
        if value is not None:
            if name == "NEG":
                unary_value = -value
            elif name == "ABS":
                unary_value = abs(value)
            else:
                unary_value = 0.0 if value == 0.0 else 1.0 if value > 0.0 else -1.0
            replacement = _constant(unary_value)
            current = _Expression("operator", name, expression.params, children)
            if replacement is not None and replacement != current:
                rules.append(f"{name.lower()}_constant")
                return replacement, tuple(rules)

        if name == "NEG" and child.name == "NEG":
            rules.append("double_neg")
            return child.children[0], tuple(rules)
        if name == "ABS" and child.name == "ABS":
            rules.append("double_abs")
            return child, tuple(rules)
        if name == "SIGN" and child.name == "SIGN":
            rules.append("double_sign")
            return child, tuple(rules)
        return _Expression("operator", name, expression.params, children), tuple(rules)

    if name in BINARY_OPERATORS:
        left, right = children
        left_constant = _constant_value(left)
        right_constant = _constant_value(right)
        if left_constant is not None and right_constant is not None:
            result = {
                "ADD": left_constant + right_constant,
                "SUB": left_constant - right_constant,
                "MUL": left_constant * right_constant,
            }[name]
            replacement = _constant(result)
            if replacement is not None:
                rules.append("constant_fold")
                return replacement, tuple(rules)

        if name == "ADD":
            if left_constant == 0.0:
                rules.append("add_zero")
                return right, tuple(rules)
            if right_constant == 0.0:
                rules.append("add_zero")
                return left, tuple(rules)
        elif name == "SUB" and right_constant == 0.0:
            rules.append("sub_zero")
            return left, tuple(rules)
        elif name == "MUL":
            if left_constant == 1.0:
                rules.append("mul_one")
                return right, tuple(rules)
            if right_constant == 1.0:
                rules.append("mul_one")
                return left, tuple(rules)

        if name in {"ADD", "MUL"} and _expression_key(right) < _expression_key(left):
            rules.append("commutative_order")
            left, right = right, left
        return _Expression("operator", name, expression.params, (left, right)), tuple(rules)

    return _Expression(expression.kind, name, expression.params, children), tuple(rules)


def display_formula_metrics(token_names: Iterable[str]) -> dict[str, Any]:
    """Return the non-destructive display form used by curated exports."""
    raw_expression = _parse_expression(token_names)
    simplified_expression, rules = _simplify_expression(raw_expression)
    return {
        "raw_formula_text": raw_expression.text(),
        "simplified_formula_text": simplified_expression.text(),
        "simplified_token_len": simplified_expression.token_len(),
        "simplification_rules": list(rules),
    }


def _load_attempts(run_dir: Path) -> tuple[list[_FormulaRow], dict[str, Any]]:
    attempts_path = run_dir / "attempts.jsonl"
    best: dict[str, dict[str, Any]] = {}
    first_attempt: dict[str, int] = {}
    attempt_counts: dict[str, int] = {}
    with attempts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            reward_value = payload.get("reward")
            if reward_value is None:
                continue
            reward = float(reward_value)
            if not math.isfinite(reward):
                continue
            formula_hash = str(payload.get("canonical_hash", ""))
            token_names = tuple(str(item) for item in payload.get("token_names", ()))
            if not formula_hash or not token_names:
                raise RuntimeError("Valid attempt lacks canonical hash or token names")
            attempt_index = int(payload["attempt_index"])
            first_attempt.setdefault(formula_hash, attempt_index)
            attempt_counts[formula_hash] = attempt_counts.get(formula_hash, 0) + 1
            current = best.get(formula_hash)
            candidate = {
                "formula_hash": formula_hash,
                "token_names": token_names,
                "reward": reward,
                "best_attempt_index": attempt_index,
            }
            if current is None or (
                reward,
                -len(token_names),
                -attempt_index,
            ) > (
                float(current["reward"]),
                -len(current["token_names"]),
                -int(current["best_attempt_index"]),
            ):
                best[formula_hash] = candidate

    rows: list[_FormulaRow] = []
    for formula_hash, payload in best.items():
        raw_expression = _parse_expression(payload["token_names"])
        simplified_expression, rules = _simplify_expression(raw_expression)
        rows.append(
            _FormulaRow(
                formula_hash=formula_hash,
                token_names=payload["token_names"],
                reward=float(payload["reward"]),
                best_attempt_index=int(payload["best_attempt_index"]),
                first_attempt_index=first_attempt[formula_hash],
                attempt_count=attempt_counts[formula_hash],
                raw_expression=raw_expression,
                simplified_expression=simplified_expression,
                simplification_rules=rules,
            )
        )

    rows.sort(key=lambda row: (-row.reward, row.raw_token_len, row.formula_hash))
    for index, row in enumerate(rows, start=1):
        if row.raw_token_len != len(row.token_names):
            raise RuntimeError("Formula token length mismatch")

    summary = json.loads(
        (run_dir / "training_summary.json").read_text(encoding="utf-8")
    )
    return rows, summary


def _select_rows(
    rows: list[_FormulaRow], size: int, tolerance: float
) -> list[tuple[_FormulaRow, str]]:
    remaining = list(rows)
    selected: list[tuple[_FormulaRow, str]] = []
    while remaining and len(selected) < size:
        highest_reward = remaining[0].reward
        eligible = [
            row for row in remaining if highest_reward - row.reward <= tolerance
        ]
        if tolerance == 0.0:
            chosen = min(
                eligible,
                key=lambda row: (row.raw_token_len, -row.reward, row.formula_hash),
            )
        else:
            chosen = min(
                eligible,
                key=lambda row: (
                    row.simplified_token_len,
                    row.raw_token_len,
                    -row.reward,
                    row.formula_hash,
                ),
            )
        raw_first = eligible[0]
        reason = "reward_order" if chosen.formula_hash == raw_first.formula_hash else "near_tie_shorter"
        selected.append((chosen, reason))
        remaining.remove(chosen)
    if len(selected) != size:
        raise RuntimeError(f"Only {len(selected)} formulas available for top {size}")
    return selected


def _tolerance_label(tolerance: float) -> str:
    if tolerance == 0.0:
        return "0"
    return f"{tolerance:.0e}".replace("+", "")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _preview_payload(
    row: _FormulaRow,
    *,
    rank: int,
    raw_rank: int,
    tolerance: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": CURATION_PREVIEW_SCHEMA_VERSION,
        "curation_rule_version": CURATION_RULE_VERSION,
        "rank": rank,
        "raw_reward_rank": raw_rank,
        "selection_reason": reason,
        "reward_tolerance": tolerance,
        "formula_hash": row.formula_hash,
        "reward": row.reward,
        "raw_formula_text": row.raw_expression.text(),
        "simplified_formula_text": row.simplified_expression.text(),
        "token_names": list(row.token_names),
        "token_len": row.raw_token_len,
        "simplified_token_len": row.simplified_token_len,
        "simplification_rules": list(row.simplification_rules),
        "best_attempt_index": row.best_attempt_index,
        "first_attempt_index": row.first_attempt_index,
        "attempt_count": row.attempt_count,
    }


def _scenario_summary(
    selected: list[tuple[_FormulaRow, str]],
    *,
    rows: list[_FormulaRow],
    size: int,
    tolerance: float,
) -> dict[str, Any]:
    selected_rows = [row for row, _ in selected]
    raw_rows = rows[:size]
    raw_reward_sum = sum(row.reward for row in raw_rows)
    selected_reward_sum = sum(row.reward for row in selected_rows)
    return {
        "size": size,
        "reward_tolerance": tolerance,
        "near_tie_shorter_count": sum(
            reason == "near_tie_shorter" for _, reason in selected
        ),
        "max_length_count": sum(row.raw_token_len == 15 for row in selected_rows),
        "raw_token_len_le_6_count": sum(row.raw_token_len <= 6 for row in selected_rows),
        "simplified_token_len_le_6_count": sum(
            row.simplified_token_len <= 6 for row in selected_rows
        ),
        "raw_token_len_le_10_count": sum(row.raw_token_len <= 10 for row in selected_rows),
        "simplified_token_len_le_10_count": sum(
            row.simplified_token_len <= 10 for row in selected_rows
        ),
        "mean_raw_token_len": sum(row.raw_token_len for row in selected_rows) / size,
        "mean_simplified_token_len": sum(
            row.simplified_token_len for row in selected_rows
        )
        / size,
        "best_reward": selected_rows[0].reward,
        "worst_reward": min(row.reward for row in selected_rows),
        "reward_sum_difference_vs_raw_top_n": selected_reward_sum - raw_reward_sum,
        "raw_ranks": [rows.index(row) + 1 for row in selected_rows],
    }


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir or run_dir / "formula_library" / "curation_preview"
    ).resolve()
    tolerances = _parse_tolerances(args.tolerances)
    rows, training_summary = _load_attempts(run_dir)
    raw_rank = {row.formula_hash: index for index, row in enumerate(rows, start=1)}

    scenario_files: dict[str, dict[str, str]] = {}
    scenarios: list[dict[str, Any]] = []
    for tolerance in tolerances:
        label = _tolerance_label(tolerance)
        scenario_files[label] = {}
        for size in (30, 50):
            selected = _select_rows(rows, size, tolerance)
            payloads = [
                _preview_payload(
                    row,
                    rank=rank,
                    raw_rank=raw_rank[row.formula_hash],
                    tolerance=tolerance,
                    reason=reason,
                )
                for rank, (row, reason) in enumerate(selected, start=1)
            ]
            path = output_dir / f"top{size}_reward_tolerance_{label}.jsonl"
            _write_jsonl(path, payloads)
            scenario_files[label][str(size)] = str(path)
            scenarios.append(
                _scenario_summary(
                    selected, rows=rows, size=size, tolerance=tolerance
                )
            )

    analysis = {
        "schema_version": CURATION_PREVIEW_SCHEMA_VERSION,
        "curation_rule_version": CURATION_RULE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_id": training_summary.get("run_id"),
            "protocol_id": training_summary.get("protocol_id"),
            "attempt_count": int(training_summary.get("attempt_count", 0)),
            "canonical_valid_formula_count": len(rows),
            "existing_top_n_artifacts_unchanged": True,
        },
        "display_simplification_rules": [
            "x+0 -> x",
            "x-0 -> x",
            "x*1 -> x",
            "double NEG/ABS/SIGN removal",
            "ABS/SIGN of CONST_0 or CONST_1",
            "safe unary combinations of CONST_0 and CONST_1",
            "constant folding only when result is representable by the token grammar",
            "ADD/MUL child ordering",
        ],
        "explicitly_not_applied": [
            "x*0 -> 0, because NaN-preserving VM semantics forbid it",
            "training reward changes",
            "canonical ledger hash changes",
            "validation or final metrics",
        ],
        "tolerances": list(tolerances),
        "scenario_files": scenario_files,
        "scenarios": scenarios,
    }
    _write_json(output_dir / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "canonical_valid_formula_count": len(rows),
                "analysis": str(output_dir / "analysis.json"),
                "scenario_files": scenario_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()