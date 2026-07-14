"""Typed parameter grammar and compiler for V3A formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from alpha_etf.research_v3a.factors import FACTOR_NAMES, WINDOWS


VOCAB_VERSION = "etf-v3a-formula-v1"
GRAMMAR_VERSION = "etf-v3a-grammar-v1"
MAX_FORMULA_TOKENS = 15
WINDOW_TOKEN_VALUES = (1, 5, 10, 20, 40, 60)
FIXED_FACTORS = ("DAYRET", "GAP", "INTRADAY", "RANGE", "CLV")
PARAMETER_FACTORS = ("ROC", "PRICE_MA", "VOL", "TS_RANK", "RSV", "MA_RATIO")
OPERATOR_NAMES = ("ADD", "SUB", "MUL", "NEG", "ABS", "SIGN", "REF", "MEAN")


@dataclass(frozen=True)
class FormulaToken:
    token_id: int
    name: str
    kind: str
    arity: int = 0
    value: float | int | None = None


TOKEN_NAMES = (
    "DAYRET",
    "GAP",
    "INTRADAY",
    "RANGE",
    "CLV",
    "ROC",
    "PRICE_MA",
    "VOL",
    "TS_RANK",
    "RSV",
    "MA_RATIO",
    "WIN_1",
    "WIN_5",
    "WIN_10",
    "WIN_20",
    "WIN_40",
    "WIN_60",
    "ADD",
    "SUB",
    "MUL",
    "NEG",
    "ABS",
    "SIGN",
    "REF",
    "MEAN",
    "CONST_0",
    "CONST_1",
)


def _make_tokens() -> tuple[FormulaToken, ...]:
    tokens: list[FormulaToken] = []
    for token_id, name in enumerate(TOKEN_NAMES):
        if name in FIXED_FACTORS:
            token = FormulaToken(token_id, name, "fixed_factor")
        elif name in PARAMETER_FACTORS:
            token = FormulaToken(token_id, name, "parameter_factor")
        elif name.startswith("WIN_"):
            token = FormulaToken(token_id, name, "window", value=int(name.removeprefix("WIN_")))
        elif name in {"ADD", "SUB", "MUL"}:
            token = FormulaToken(token_id, name, "operator", arity=2)
        elif name in {"NEG", "ABS", "SIGN"}:
            token = FormulaToken(token_id, name, "operator", arity=1)
        elif name in {"REF", "MEAN"}:
            token = FormulaToken(token_id, name, "rolling_operator", arity=2)
        elif name == "CONST_0":
            token = FormulaToken(token_id, name, "constant", value=0.0)
        elif name == "CONST_1":
            token = FormulaToken(token_id, name, "constant", value=1.0)
        else:  # pragma: no cover - frozen token list.
            raise AssertionError(f"Unknown token: {name}")
        tokens.append(token)
    return tuple(tokens)


TOKENS = _make_tokens()


@dataclass(frozen=True)
class FormulaVocab:
    tokens: tuple[FormulaToken, ...] = TOKENS
    version: str = VOCAB_VERSION

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def token_names(self) -> tuple[str, ...]:
        return tuple(token.name for token in self.tokens)

    def id_to_token(self, token_id: int) -> FormulaToken:
        if token_id < 0 or token_id >= self.size:
            raise KeyError(f"Unknown V3A token id: {token_id}")
        return self.tokens[token_id]

    def name_to_id(self, name: str) -> int:
        try:
            return self.token_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown V3A token name: {name}") from exc

    def encode(self, names: Iterable[str]) -> list[int]:
        return [self.name_to_id(name) for name in names]

    def decode(self, token_ids: Iterable[int]) -> list[str]:
        return [self.id_to_token(int(token_id)).name for token_id in token_ids]

    def to_config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "grammar_version": GRAMMAR_VERSION,
            "token_names": list(self.token_names),
            "max_formula_tokens": MAX_FORMULA_TOKENS,
            "policy_special_tokens": {"PAD": 0, "BOS": 1, "EOS": 2},
        }


FORMULA_VOCAB = FormulaVocab()


@dataclass(frozen=True)
class Expression:
    kind: str
    name: str
    params: tuple[int | float, ...] = ()
    children: tuple["Expression", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "params": list(self.params),
            "children": [child.to_dict() for child in self.children],
        }

    def text(self) -> str:
        if self.kind == "factor":
            args = ",".join(str(item) for item in self.params)
            return self.name if not args else f"{self.name}({args})"
        if self.kind == "constant":
            return self.name
        args = [child.text() for child in self.children]
        args.extend(str(item) for item in self.params)
        return f"{self.name}({','.join(args)})"


@dataclass(frozen=True)
class WindowValue:
    value: int


StackValue = Expression | WindowValue


@dataclass(frozen=True)
class GrammarState:
    stack: tuple[str, ...] = ()
    pending_factor: str = ""
    pending_windows: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.pending_factor and self.stack == ("value",)


def _allowed_pending_windows(state: GrammarState) -> tuple[int, ...]:
    if not state.pending_factor:
        return ()
    if state.pending_factor != "MA_RATIO":
        return WINDOWS
    if not state.pending_windows:
        return WINDOWS[:-1]
    short = state.pending_windows[0]
    return tuple(window for window in WINDOWS if window > short)


def transition_state(state: GrammarState, token: FormulaToken) -> GrammarState:
    if state.pending_factor:
        if token.kind != "window" or int(token.value) not in _allowed_pending_windows(state):
            raise ValueError(f"Token {token.name} is invalid for pending {state.pending_factor}")
        windows = state.pending_windows + (int(token.value),)
        required = 2 if state.pending_factor == "MA_RATIO" else 1
        if len(windows) < required:
            return GrammarState(state.stack, state.pending_factor, windows)
        return GrammarState(state.stack + ("value",))

    stack = list(state.stack)
    if stack and stack[-1].startswith("window:") and token.kind != "rolling_operator":
        raise ValueError(f"Only REF/MEAN may follow a window, got {token.name}")

    if token.kind in {"fixed_factor", "constant"}:
        stack.append("value")
        return GrammarState(tuple(stack))
    if token.kind == "parameter_factor":
        return GrammarState(tuple(stack), token.name, ())
    if token.kind == "window":
        if not stack or stack[-1] != "value":
            raise ValueError(f"Window {token.name} requires a value expression")
        return GrammarState(tuple(stack + [f"window:{int(token.value)}"]))
    if token.kind == "operator":
        if len(stack) < token.arity or any(item != "value" for item in stack[-token.arity :]):
            raise ValueError(f"Operator {token.name} has invalid operands")
        del stack[-token.arity :]
        stack.append("value")
        return GrammarState(tuple(stack))
    if token.kind == "rolling_operator":
        if len(stack) < 2 or stack[-2] != "value" or not stack[-1].startswith("window:"):
            raise ValueError(f"Rolling operator {token.name} requires value and window")
        window = int(stack[-1].split(":", 1)[1])
        if token.name == "MEAN" and window == 1:
            raise ValueError("MEAN does not allow WIN_1")
        del stack[-2:]
        stack.append("value")
        return GrammarState(tuple(stack))
    raise ValueError(f"Unsupported token kind: {token.kind}")


def minimum_tokens_to_finish(state: GrammarState) -> int:
    stack = list(state.stack)
    pending = 0
    if state.pending_factor:
        required = 2 if state.pending_factor == "MA_RATIO" else 1
        pending = required - len(state.pending_windows)
        stack.append("value")
    if stack and stack[-1].startswith("window:"):
        pending += 1
        stack[-2:] = ["value"]
    if any(item != "value" for item in stack):
        return 10**9
    return pending + max(len(stack) - 1, 0)


def allowed_formula_token_ids(
    state: GrammarState,
    *,
    formula_length: int,
    max_length: int = MAX_FORMULA_TOKENS,
    vocab: FormulaVocab = FORMULA_VOCAB,
) -> tuple[int, ...]:
    if formula_length >= max_length:
        return ()
    candidates: list[int] = []
    remaining_after = max_length - formula_length - 1
    for token in vocab.tokens:
        try:
            next_state = transition_state(state, token)
        except (ValueError, AssertionError):
            continue
        if minimum_tokens_to_finish(next_state) <= remaining_after:
            candidates.append(token.token_id)
    return tuple(candidates)


class ExpressionBuilder:
    def __init__(self, vocab: FormulaVocab = FORMULA_VOCAB):
        self.vocab = vocab
        self.stack: list[StackValue] = []
        self.pending_factor = ""
        self.pending_windows: list[int] = []

    @property
    def state(self) -> GrammarState:
        stack = tuple(
            "value" if isinstance(item, Expression) else f"window:{item.value}"
            for item in self.stack
        )
        return GrammarState(stack, self.pending_factor, tuple(self.pending_windows))

    @property
    def is_valid(self) -> bool:
        return not self.pending_factor and len(self.stack) == 1 and isinstance(self.stack[0], Expression)

    def add_token_id(self, token_id: int) -> None:
        self.add_token(self.vocab.id_to_token(int(token_id)))

    def add_token(self, token: FormulaToken) -> None:
        transition_state(self.state, token)
        if self.pending_factor:
            window = int(token.value)
            self.pending_windows.append(window)
            required = 2 if self.pending_factor == "MA_RATIO" else 1
            if len(self.pending_windows) == required:
                self.stack.append(
                    Expression("factor", self.pending_factor, tuple(self.pending_windows))
                )
                self.pending_factor = ""
                self.pending_windows = []
            return

        if token.kind == "fixed_factor":
            self.stack.append(Expression("factor", token.name))
        elif token.kind == "parameter_factor":
            self.pending_factor = token.name
            self.pending_windows = []
        elif token.kind == "constant":
            self.stack.append(Expression("constant", token.name, (float(token.value),)))
        elif token.kind == "window":
            self.stack.append(WindowValue(int(token.value)))
        elif token.kind == "operator":
            children = tuple(self.stack[-token.arity :])
            if not all(isinstance(item, Expression) for item in children):
                raise ValueError(f"Operator {token.name} received non-expression operand")
            del self.stack[-token.arity :]
            self.stack.append(Expression("operator", token.name, children=children))
        elif token.kind == "rolling_operator":
            expr, window = self.stack[-2:]
            if not isinstance(expr, Expression) or not isinstance(window, WindowValue):
                raise ValueError(f"Rolling operator {token.name} received invalid operands")
            del self.stack[-2:]
            self.stack.append(
                Expression("operator", token.name, params=(window.value,), children=(expr,))
            )
        else:  # pragma: no cover - frozen token kinds.
            raise ValueError(f"Unsupported token kind: {token.kind}")

    def get_expression(self) -> Expression:
        if not self.is_valid:
            raise ValueError(f"Formula is incomplete: state={self.state}")
        value = self.stack[0]
        assert isinstance(value, Expression)
        return value


def parse_formula(
    token_ids: Iterable[int], vocab: FormulaVocab = FORMULA_VOCAB
) -> Expression:
    ids = [int(item) for item in token_ids]
    if not ids:
        raise ValueError("Formula must contain at least one token")
    if len(ids) > MAX_FORMULA_TOKENS:
        raise ValueError(f"Formula exceeds {MAX_FORMULA_TOKENS} tokens")
    builder = ExpressionBuilder(vocab)
    for token_id in ids:
        builder.add_token_id(token_id)
    return builder.get_expression()


FACTOR_CODE_OFFSET = 0
CONST_0_CODE = len(FACTOR_NAMES)
CONST_1_CODE = CONST_0_CODE + 1
ADD_CODE = CONST_1_CODE + 1
SUB_CODE = ADD_CODE + 1
MUL_CODE = SUB_CODE + 1
NEG_CODE = MUL_CODE + 1
ABS_CODE = NEG_CODE + 1
SIGN_CODE = ABS_CODE + 1
REF_CODE_BY_WINDOW = {
    window: SIGN_CODE + 1 + index for index, window in enumerate(WINDOW_TOKEN_VALUES)
}
MEAN_CODE_BY_WINDOW = {
    window: max(REF_CODE_BY_WINDOW.values()) + 1 + index for index, window in enumerate(WINDOWS)
}
COMPILED_VOCAB_SIZE = max(MEAN_CODE_BY_WINDOW.values()) + 1


def transition_instruction_code(
    state: GrammarState, token: FormulaToken
) -> int | None:
    """Return the VM instruction emitted by one valid grammar transition."""

    transition_state(state, token)
    if state.pending_factor:
        windows = state.pending_windows + (int(token.value),)
        required = 2 if state.pending_factor == "MA_RATIO" else 1
        if len(windows) < required:
            return None
        name = state.pending_factor + "_" + "_".join(str(value) for value in windows)
        return FACTOR_NAMES.index(name)
    if token.kind == "fixed_factor":
        return FACTOR_NAMES.index(token.name)
    if token.kind == "constant":
        return CONST_0_CODE if token.name == "CONST_0" else CONST_1_CODE
    if token.kind in {"parameter_factor", "window"}:
        return None
    if token.name == "ADD":
        return ADD_CODE
    if token.name == "SUB":
        return SUB_CODE
    if token.name == "MUL":
        return MUL_CODE
    if token.name == "NEG":
        return NEG_CODE
    if token.name == "ABS":
        return ABS_CODE
    if token.name == "SIGN":
        return SIGN_CODE
    window = int(state.stack[-1].split(":", 1)[1])
    if token.name == "REF":
        return REF_CODE_BY_WINDOW[window]
    if token.name == "MEAN":
        return MEAN_CODE_BY_WINDOW[window]
    raise ValueError(f"Unsupported V3A instruction transition: {token.name}")


@dataclass(frozen=True)
class CompiledFormula:
    source_token_ids: tuple[int, ...]
    expression: Expression
    instructions: tuple[int, ...]


def _factor_instance_name(expression: Expression) -> str:
    if not expression.params:
        return expression.name
    return expression.name + "_" + "_".join(str(item) for item in expression.params)


def _compile_expression(expression: Expression, output: list[int]) -> None:
    if expression.kind == "factor":
        name = _factor_instance_name(expression)
        try:
            output.append(FACTOR_NAMES.index(name))
        except ValueError as exc:
            raise ValueError(f"Unknown V3A factor instance: {name}") from exc
        return
    if expression.kind == "constant":
        output.append(CONST_0_CODE if expression.name == "CONST_0" else CONST_1_CODE)
        return
    for child in expression.children:
        _compile_expression(child, output)
    if expression.name == "ADD":
        output.append(ADD_CODE)
    elif expression.name == "SUB":
        output.append(SUB_CODE)
    elif expression.name == "MUL":
        output.append(MUL_CODE)
    elif expression.name == "NEG":
        output.append(NEG_CODE)
    elif expression.name == "ABS":
        output.append(ABS_CODE)
    elif expression.name == "SIGN":
        output.append(SIGN_CODE)
    elif expression.name == "REF":
        output.append(REF_CODE_BY_WINDOW[int(expression.params[0])])
    elif expression.name == "MEAN":
        output.append(MEAN_CODE_BY_WINDOW[int(expression.params[0])])
    else:  # pragma: no cover - frozen operators.
        raise ValueError(f"Unknown V3A operator: {expression.name}")


def compile_formula(
    token_ids: Iterable[int], vocab: FormulaVocab = FORMULA_VOCAB
) -> CompiledFormula:
    ids = tuple(int(item) for item in token_ids)
    expression = parse_formula(ids, vocab)
    instructions: list[int] = []
    _compile_expression(expression, instructions)
    return CompiledFormula(ids, expression, tuple(instructions))