"""Action masking and formula generation for the typed V3A grammar."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch
from torch.distributions import Categorical

from alpha_etf.research_v3a.language import (
    FORMULA_VOCAB,
    MAX_FORMULA_TOKENS,
    FormulaVocab,
    GrammarState,
    allowed_formula_token_ids,
    transition_state,
)


NEG_INF = -1.0e9


@dataclass(frozen=True)
class PolicyVocab:
    formula_vocab: FormulaVocab = FORMULA_VOCAB

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def formula_offset(self) -> int:
        return 3

    @property
    def size(self) -> int:
        return self.formula_offset + self.formula_vocab.size

    @property
    def token_names(self) -> tuple[str, ...]:
        return ("PAD", "BOS", "EOS") + self.formula_vocab.token_names

    @property
    def special_tokens(self) -> dict[str, int]:
        return {"PAD": self.pad_id, "BOS": self.bos_id, "EOS": self.eos_id}

    def to_model_id(self, formula_token_id: int) -> int:
        self.formula_vocab.id_to_token(int(formula_token_id))
        return int(formula_token_id) + self.formula_offset

    def to_formula_id(self, model_token_id: int) -> int:
        formula_id = int(model_token_id) - self.formula_offset
        self.formula_vocab.id_to_token(formula_id)
        return formula_id

    def is_formula_id(self, model_token_id: int) -> bool:
        return self.formula_offset <= int(model_token_id) < self.size


@dataclass(frozen=True)
class SamplingConfig:
    max_len: int = MAX_FORMULA_TOKENS

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "random_action_distribution": "uniform_over_legal_policy_actions",
            "eos_rule": "same_legal_eos_action_as_policy",
        }


@dataclass
class SampleBatch:
    formulas: list[list[int]]
    model_sequences: list[list[int]]
    log_prob_sums: torch.Tensor
    entropy_sums: torch.Tensor
    normalized_entropy_sums: torch.Tensor
    formula_lengths: torch.Tensor
    avg_allowed_actions: float


def allowed_policy_action_ids(
    state: GrammarState,
    *,
    formula_length: int,
    policy_vocab: PolicyVocab,
    config: SamplingConfig,
) -> tuple[int, ...]:
    actions: list[int] = []
    if state.valid:
        actions.append(policy_vocab.eos_id)
    actions.extend(
        policy_vocab.to_model_id(token_id)
        for token_id in allowed_formula_token_ids(
            state,
            formula_length=formula_length,
            max_length=config.max_len,
            vocab=policy_vocab.formula_vocab,
        )
    )
    return tuple(actions)


def build_action_mask(
    states: list[GrammarState],
    formula_lengths: torch.Tensor,
    done: torch.Tensor,
    policy_vocab: PolicyVocab,
    config: SamplingConfig,
) -> torch.Tensor:
    batch_size = len(states)
    if formula_lengths.shape != (batch_size,) or done.shape != (batch_size,):
        raise ValueError("V3A action-mask state shapes differ")
    mask = torch.full((batch_size, policy_vocab.size), NEG_INF, device=formula_lengths.device)
    for row, state in enumerate(states):
        if bool(done[row].item()):
            mask[row, policy_vocab.pad_id] = 0.0
            continue
        length = int(formula_lengths[row].item())
        for model_id in allowed_policy_action_ids(
            state,
            formula_length=length,
            policy_vocab=policy_vocab,
            config=config,
        ):
            mask[row, model_id] = 0.0
    return mask


def sample_formulas(
    model: torch.nn.Module,
    batch_size: int,
    policy_vocab: PolicyVocab,
    config: SamplingConfig,
    device: torch.device,
) -> SampleBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.max_len < 1 or config.max_len > MAX_FORMULA_TOKENS:
        raise ValueError(f"max_len must be in [1,{MAX_FORMULA_TOKENS}]")

    inp = torch.full((batch_size, 1), policy_vocab.bos_id, dtype=torch.long, device=device)
    states = [GrammarState() for _ in range(batch_size)]
    formulas: list[list[int]] = [[] for _ in range(batch_size)]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    log_prob_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    entropy_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    normalized_entropy_sums = torch.zeros(
        batch_size, dtype=torch.float32, device=device
    )
    allowed_counts: list[float] = []

    for _ in range(config.max_len + 1):
        logits, _ = model(inp)
        action_mask = build_action_mask(states, lengths, done, policy_vocab, config)
        allowed_counts.append(float((action_mask == 0.0).sum(dim=1).float().mean().item()))
        dist = Categorical(logits=logits + action_mask)
        action = dist.sample()
        log_prob_sums = log_prob_sums + dist.log_prob(action)
        step_entropy = dist.entropy()
        entropy_sums = entropy_sums + step_entropy
        allowed_per_row = (action_mask == 0.0).sum(dim=1)
        normalizer = torch.log(allowed_per_row.clamp_min(2).to(step_entropy.dtype))
        normalized_entropy_sums = normalized_entropy_sums + torch.where(
            allowed_per_row > 1,
            step_entropy / normalizer,
            torch.zeros_like(step_entropy),
        )

        for row, model_id in enumerate(action.detach().cpu().tolist()):
            if bool(done[row].item()):
                continue
            if model_id == policy_vocab.eos_id:
                done[row] = True
                continue
            formula_id = policy_vocab.to_formula_id(model_id)
            formulas[row].append(formula_id)
            states[row] = transition_state(
                states[row], policy_vocab.formula_vocab.id_to_token(formula_id)
            )
            lengths[row] += 1

        inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
        if bool(done.all().item()):
            break
    if not bool(done.all().item()):
        raise RuntimeError(f"V3A sampler left {int((~done).sum().item())} formulas unfinished")
    if not all(state.valid for state in states):
        raise RuntimeError("V3A sampler ended with an invalid grammar state")
    return SampleBatch(
        formulas=formulas,
        model_sequences=inp.detach().cpu().tolist(),
        log_prob_sums=log_prob_sums,
        entropy_sums=entropy_sums,
        normalized_entropy_sums=normalized_entropy_sums,
        formula_lengths=lengths.detach().cpu(),
        avg_allowed_actions=float(sum(allowed_counts) / len(allowed_counts))
        if allowed_counts
        else math.nan,
    )


def generate_random_formula(
    rng: random.Random,
    *,
    config: SamplingConfig = SamplingConfig(),
    policy_vocab: PolicyVocab = PolicyVocab(),
) -> list[int]:
    state = GrammarState()
    formula: list[int] = []
    while True:
        allowed = allowed_policy_action_ids(
            state,
            formula_length=len(formula),
            policy_vocab=policy_vocab,
            config=config,
        )
        if not allowed:
            raise RuntimeError(f"Random V3A generation reached a dead end: {state}")
        model_id = int(rng.choice(allowed))
        if model_id == policy_vocab.eos_id:
            return formula
        token_id = policy_vocab.to_formula_id(model_id)
        formula.append(token_id)
        state = transition_state(state, policy_vocab.formula_vocab.id_to_token(token_id))