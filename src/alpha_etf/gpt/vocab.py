"""Phase 3a token vocabulary for ETF formula experiments."""

from __future__ import annotations

from dataclasses import dataclass


VOCAB_VERSION = "phase3a-v1"


@dataclass(frozen=True)
class TokenSpec:
    name: str
    kind: str
    arity: int = 0
    source: str = ""
    value: float | None = None


FEATURE_TOKENS: tuple[TokenSpec, ...] = (
    TokenSpec("open", "feature", source="qfq"),
    TokenSpec("high", "feature", source="qfq"),
    TokenSpec("low", "feature", source="qfq"),
    TokenSpec("close", "feature", source="qfq"),
    TokenSpec("volume", "feature", source="raw"),
    TokenSpec("amount", "feature", source="raw"),
)

CONSTANT_TOKENS: tuple[TokenSpec, ...] = (
    TokenSpec("CONST_0", "constant", value=0.0),
    TokenSpec("CONST_1", "constant", value=1.0),
)

OPERATOR_TOKENS: tuple[TokenSpec, ...] = (
    TokenSpec("ADD", "operator", arity=2),
    TokenSpec("SUB", "operator", arity=2),
    TokenSpec("MUL", "operator", arity=2),
    TokenSpec("DIV", "operator", arity=2),
    TokenSpec("NEG", "operator", arity=1),
    TokenSpec("ABS", "operator", arity=1),
    TokenSpec("SIGN", "operator", arity=1),
    TokenSpec("DELAY1", "operator", arity=1),
    TokenSpec("DELAY5", "operator", arity=1),
    TokenSpec("DELAY10", "operator", arity=1),
    TokenSpec("MA5", "operator", arity=1),
    TokenSpec("MA10", "operator", arity=1),
    TokenSpec("MA20", "operator", arity=1),
    TokenSpec("STD10", "operator", arity=1),
    TokenSpec("RET5", "operator", arity=1),
    TokenSpec("RET10", "operator", arity=1),
    TokenSpec("DECAY", "operator", arity=1),
)


@dataclass(frozen=True)
class FormulaVocab:
    tokens: tuple[TokenSpec, ...]
    version: str = VOCAB_VERSION

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def token_names(self) -> tuple[str, ...]:
        return tuple(token.name for token in self.tokens)

    @property
    def input_ids(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.tokens) if token.kind in {"feature", "constant"})

    @property
    def unary_operator_ids(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.tokens) if token.kind == "operator" and token.arity == 1)

    @property
    def binary_operator_ids(self) -> tuple[int, ...]:
        return tuple(i for i, token in enumerate(self.tokens) if token.kind == "operator" and token.arity == 2)

    def id_to_token(self, token_id: int) -> TokenSpec:
        if token_id < 0 or token_id >= self.size:
            raise KeyError(f"Unknown token id: {token_id}")
        return self.tokens[token_id]

    def id_to_name(self, token_id: int) -> str:
        return self.id_to_token(token_id).name

    def name_to_id(self, name: str) -> int:
        for i, token in enumerate(self.tokens):
            if token.name == name:
                return i
        raise KeyError(f"Unknown token name: {name}")

    def encode(self, token_names: list[str] | tuple[str, ...]) -> list[int]:
        return [self.name_to_id(name) for name in token_names]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.id_to_name(int(token_id)) for token_id in token_ids]


FORMULA_VOCAB = FormulaVocab(FEATURE_TOKENS + CONSTANT_TOKENS + OPERATOR_TOKENS)
