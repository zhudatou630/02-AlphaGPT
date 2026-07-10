"""Price-only formula vocabulary for the V2 ETF research line."""

from __future__ import annotations

from alpha_etf.gpt.vocab import CONSTANT_TOKENS, OPERATOR_TOKENS, FormulaVocab, TokenSpec


VOCAB_VERSION_V2 = "price-event-v2"

PRICE_FEATURE_TOKENS: tuple[TokenSpec, ...] = (
    TokenSpec("open", "feature", source="qfq"),
    TokenSpec("high", "feature", source="qfq"),
    TokenSpec("low", "feature", source="qfq"),
    TokenSpec("close", "feature", source="qfq"),
)

FORMULA_VOCAB_V2 = FormulaVocab(
    PRICE_FEATURE_TOKENS + CONSTANT_TOKENS + OPERATOR_TOKENS,
    version=VOCAB_VERSION_V2,
)
