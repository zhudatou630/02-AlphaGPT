"""Action masking and sampling for Phase 3b formula policies."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch.distributions import Categorical

from alpha_etf.gpt.vocab import FORMULA_VOCAB, FormulaVocab


NEG_INF = -1.0e9


@dataclass(frozen=True)
class PolicyVocab:
    formula_vocab: FormulaVocab = FORMULA_VOCAB
    pad_name: str = "PAD"
    bos_name: str = "BOS"
    eos_name: str = "EOS"

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
        return (self.pad_name, self.bos_name, self.eos_name) + self.formula_vocab.token_names

    @property
    def special_tokens(self) -> dict[str, int]:
        return {self.pad_name: self.pad_id, self.bos_name: self.bos_id, self.eos_name: self.eos_id}

    def to_model_id(self, formula_token_id: int) -> int:
        return int(formula_token_id) + self.formula_offset

    def to_formula_id(self, model_token_id: int) -> int:
        formula_token_id = int(model_token_id) - self.formula_offset
        self.formula_vocab.id_to_token(formula_token_id)
        return formula_token_id

    def is_formula_id(self, model_token_id: int) -> bool:
        return self.formula_offset <= int(model_token_id) < self.size


@dataclass(frozen=True)
class SamplingConfig:
    max_len: int = 16
    min_formula_len: int = 3


@dataclass
class SampleBatch:
    formulas: list[list[int]]
    model_sequences: list[list[int]]
    log_prob_sums: torch.Tensor
    entropy_sums: torch.Tensor
    formula_lengths: torch.Tensor
    avg_allowed_actions: float


def _min_tokens_to_finish(stack_depth: int) -> int:
    if stack_depth <= 0:
        return 1
    return stack_depth - 1


def _formula_token_allowed(
    stack_depth: int,
    formula_len: int,
    formula_token_id: int,
    vocab: FormulaVocab,
    config: SamplingConfig,
) -> bool:
    if formula_len >= config.max_len:
        return False
    token = vocab.id_to_token(formula_token_id)
    if token.kind in {"feature", "constant"}:
        depth_after = stack_depth + 1
    elif token.kind == "operator" and token.arity == 1:
        if stack_depth < 1:
            return False
        depth_after = stack_depth
    elif token.kind == "operator" and token.arity == 2:
        if stack_depth < 2:
            return False
        depth_after = stack_depth - 1
    else:
        return False

    remaining_after = config.max_len - (formula_len + 1)
    return _min_tokens_to_finish(depth_after) <= remaining_after


def build_action_mask(
    stack_depths: torch.Tensor,
    formula_lengths: torch.Tensor,
    done: torch.Tensor,
    policy_vocab: PolicyVocab,
    config: SamplingConfig,
) -> torch.Tensor:
    """Return additive logits mask where disallowed actions are NEG_INF."""

    batch_size = int(stack_depths.shape[0])
    mask = torch.full((batch_size, policy_vocab.size), NEG_INF, device=stack_depths.device)

    for row in range(batch_size):
        if bool(done[row].item()):
            mask[row, policy_vocab.pad_id] = 0.0
            continue

        depth = int(stack_depths[row].item())
        length = int(formula_lengths[row].item())

        if depth == 1 and length >= config.min_formula_len:
            mask[row, policy_vocab.eos_id] = 0.0

        for formula_token_id in range(policy_vocab.formula_vocab.size):
            if _formula_token_allowed(depth, length, formula_token_id, policy_vocab.formula_vocab, config):
                mask[row, policy_vocab.to_model_id(formula_token_id)] = 0.0

    return mask


def _stack_delta(model_token_id: int, policy_vocab: PolicyVocab) -> int:
    formula_token_id = policy_vocab.to_formula_id(model_token_id)
    token = policy_vocab.formula_vocab.id_to_token(formula_token_id)
    if token.kind in {"feature", "constant"}:
        return 1
    if token.kind == "operator":
        return 1 - token.arity
    raise ValueError(f"Unsupported token kind: {token.kind}")


def sample_formulas(
    model: torch.nn.Module,
    batch_size: int,
    policy_vocab: PolicyVocab,
    config: SamplingConfig,
    device: torch.device,
) -> SampleBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.max_len < config.min_formula_len:
        raise ValueError("max_len must be >= min_formula_len")

    inp = torch.full((batch_size, 1), policy_vocab.bos_id, dtype=torch.long, device=device)
    stack_depths = torch.zeros(batch_size, dtype=torch.long, device=device)
    formula_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    log_prob_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    entropy_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    formulas: list[list[int]] = [[] for _ in range(batch_size)]
    allowed_counts: list[float] = []

    # +1 leaves room for EOS after the last formula token.
    for _ in range(config.max_len + 1):
        logits, _ = model(inp)
        action_mask = build_action_mask(stack_depths, formula_lengths, done, policy_vocab, config)
        allowed_counts.append(float((action_mask == 0.0).sum(dim=1).float().mean().item()))
        dist = Categorical(logits=logits + action_mask)
        action = dist.sample()
        log_prob_sums = log_prob_sums + dist.log_prob(action)
        entropy_sums = entropy_sums + dist.entropy()

        for row, model_token_id in enumerate(action.detach().cpu().tolist()):
            if done[row].item():
                continue
            if model_token_id == policy_vocab.eos_id:
                done[row] = True
                continue
            if policy_vocab.is_formula_id(model_token_id):
                formula_token_id = policy_vocab.to_formula_id(model_token_id)
                formulas[row].append(formula_token_id)

        formula_action = torch.zeros(batch_size, dtype=torch.bool, device=device)
        deltas = torch.zeros(batch_size, dtype=torch.long, device=device)
        for row, model_token_id in enumerate(action.detach().cpu().tolist()):
            if done[row].item() and model_token_id != policy_vocab.eos_id:
                continue
            if policy_vocab.is_formula_id(model_token_id):
                formula_action[row] = True
                deltas[row] = _stack_delta(model_token_id, policy_vocab)

        stack_depths = stack_depths + deltas
        formula_lengths = formula_lengths + formula_action.long()
        inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
        if bool(done.all().item()):
            break

    if not bool(done.all().item()):
        # This should be unreachable when the mask is feasible. Keep fail-fast behavior.
        unfinished = int((~done).sum().item())
        raise RuntimeError(f"Sampling ended with {unfinished} unfinished formulas")

    sequences = inp.detach().cpu().tolist()
    avg_allowed = float(sum(allowed_counts) / max(len(allowed_counts), 1)) if allowed_counts else math.nan
    return SampleBatch(
        formulas=formulas,
        model_sequences=sequences,
        log_prob_sums=log_prob_sums,
        entropy_sums=entropy_sums,
        formula_lengths=formula_lengths.detach().cpu(),
        avg_allowed_actions=avg_allowed,
    )