"""Semi-controlled random token generation for Phase 3a."""

from __future__ import annotations

from dataclasses import dataclass

import random

from alpha_etf.gpt.vocab import FORMULA_VOCAB, FormulaVocab


@dataclass(frozen=True)
class RandomFormulaConfig:
    min_len: int = 3
    max_len: int = 12
    max_stack_depth: int = 4


def generate_random_formula(
    rng: random.Random,
    vocab: FormulaVocab = FORMULA_VOCAB,
    config: RandomFormulaConfig = RandomFormulaConfig(),
) -> list[int]:
    if config.min_len < 1:
        raise ValueError("min_len must be positive")
    if config.max_len < config.min_len:
        raise ValueError("max_len must be >= min_len")
    if config.max_stack_depth < 1:
        raise ValueError("max_stack_depth must be positive")

    target_len = rng.randint(config.min_len, config.max_len)
    tokens: list[int] = []
    stack_depth = 0

    for pos in range(target_len):
        remaining_after = target_len - pos - 1
        candidates: list[tuple[int, float, int]] = []

        if stack_depth == 0:
            candidates.extend((token_id, 1.0, 1) for token_id in vocab.input_ids)
        else:
            must_reduce = stack_depth - 1 >= remaining_after + 1
            if not must_reduce and stack_depth < config.max_stack_depth:
                candidates.extend((token_id, 0.35, 1) for token_id in vocab.input_ids)

            candidates.extend((token_id, 0.40, 0) for token_id in vocab.unary_operator_ids)

            if stack_depth >= 2:
                weight = 0.25
                if remaining_after <= stack_depth - 1:
                    weight = 1.00
                candidates.extend((token_id, weight, -1) for token_id in vocab.binary_operator_ids)

        if not candidates:
            candidates.extend((token_id, 1.0, 1) for token_id in vocab.input_ids)

        token_ids = [item[0] for item in candidates]
        weights = [item[1] for item in candidates]
        deltas = [item[2] for item in candidates]
        idx = rng.choices(range(len(token_ids)), weights=weights, k=1)[0]
        tokens.append(token_ids[idx])
        stack_depth += deltas[idx]

    while stack_depth > 1:
        token_id = rng.choice(vocab.binary_operator_ids)
        tokens.append(token_id)
        stack_depth -= 1

    return tokens
