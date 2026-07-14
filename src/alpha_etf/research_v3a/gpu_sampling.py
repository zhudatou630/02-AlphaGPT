"""Table-driven tensor sampling for the frozen V3A formula grammar."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from alpha_etf.research_v3a.language import (
    FORMULA_VOCAB,
    MAX_FORMULA_TOKENS,
    GrammarState,
    allowed_formula_token_ids,
    transition_instruction_code,
    transition_state,
)
from alpha_etf.research_v3a.sampling import PolicyVocab, SamplingConfig


@dataclass(frozen=True)
class TensorGrammarTables:
    states: tuple[GrammarState, ...]
    legal_actions: torch.Tensor
    next_states: torch.Tensor
    emitted_codes: torch.Tensor
    initial_state_id: int
    max_vm_stack_depth: int


@dataclass(frozen=True)
class FormulaTensorBatch:
    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    vm_codes: torch.Tensor
    vm_lengths: torch.Tensor
    log_prob_sums: torch.Tensor
    entropy_sums: torch.Tensor
    normalized_entropy_sums: torch.Tensor
    average_allowed_actions: torch.Tensor


def _state_key(state: GrammarState) -> tuple[tuple[str, ...], str, tuple[int, ...]]:
    return state.stack, state.pending_factor, state.pending_windows


def build_tensor_grammar_tables(
    *,
    device: torch.device,
    policy_vocab: PolicyVocab = PolicyVocab(),
    config: SamplingConfig = SamplingConfig(),
) -> TensorGrammarTables:
    """Enumerate reachable states once and move the frozen transition tables to a device."""

    states_by_length: list[set[GrammarState]] = [{GrammarState()}]
    all_states = {GrammarState()}
    for length in range(config.max_len):
        next_at_length: set[GrammarState] = set()
        for state in states_by_length[length]:
            for token_id in allowed_formula_token_ids(
                state,
                formula_length=length,
                max_length=config.max_len,
                vocab=policy_vocab.formula_vocab,
            ):
                next_at_length.add(
                    transition_state(state, policy_vocab.formula_vocab.id_to_token(token_id))
                )
        states_by_length.append(next_at_length)
        all_states.update(next_at_length)

    states = tuple(sorted(all_states, key=_state_key))
    state_ids = {state: index for index, state in enumerate(states)}
    legal = torch.zeros(
        (config.max_len + 1, len(states), policy_vocab.size), dtype=torch.bool
    )
    next_states = torch.full(
        (len(states), policy_vocab.formula_vocab.size), -1, dtype=torch.long
    )
    emitted = torch.full_like(next_states, -1)

    for length, reachable in enumerate(states_by_length):
        for state in reachable:
            state_id = state_ids[state]
            if state.valid:
                legal[length, state_id, policy_vocab.eos_id] = True
            for token_id in allowed_formula_token_ids(
                state,
                formula_length=length,
                max_length=config.max_len,
                vocab=policy_vocab.formula_vocab,
            ):
                token = policy_vocab.formula_vocab.id_to_token(token_id)
                next_state = transition_state(state, token)
                model_id = policy_vocab.to_model_id(token_id)
                legal[length, state_id, model_id] = True
                next_states[state_id, token_id] = state_ids[next_state]
                instruction = transition_instruction_code(state, token)
                if instruction is not None:
                    emitted[state_id, token_id] = int(instruction)

    max_vm_stack_depth = max(
        sum(item == "value" for item in state.stack)
        + (1 if state.pending_factor else 0)
        for state in states
    )
    return TensorGrammarTables(
        states=states,
        legal_actions=legal.to(device),
        next_states=next_states.to(device),
        emitted_codes=emitted.to(device),
        initial_state_id=state_ids[GrammarState()],
        max_vm_stack_depth=max_vm_stack_depth,
    )


class TensorFormulaSampler:
    def __init__(
        self,
        *,
        device: torch.device,
        policy_vocab: PolicyVocab = PolicyVocab(),
        config: SamplingConfig = SamplingConfig(),
    ):
        self.device = device
        self.policy_vocab = policy_vocab
        self.config = config
        self.tables = build_tensor_grammar_tables(
            device=device, policy_vocab=policy_vocab, config=config
        )

    def sample_policy(
        self, model: torch.nn.Module, batch_size: int
    ) -> FormulaTensorBatch:
        return self._sample(model=model, batch_size=batch_size)

    def sample_uniform(self, batch_size: int) -> FormulaTensorBatch:
        return self._sample(model=None, batch_size=batch_size)

    def _sample(
        self, *, model: torch.nn.Module | None, batch_size: int
    ) -> FormulaTensorBatch:
        if batch_size < 1:
            raise ValueError("Stage D tensor sampler batch size must be positive")
        max_len = self.config.max_len
        vocab = self.policy_vocab
        sequences = torch.full(
            (batch_size, max_len + 2),
            vocab.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        sequences[:, 0] = vocab.bos_id
        token_ids = torch.full(
            (batch_size, max_len), -1, dtype=torch.long, device=self.device
        )
        vm_codes = torch.full_like(token_ids, -1)
        token_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        vm_lengths = torch.zeros_like(token_lengths)
        state_ids = torch.full(
            (batch_size,),
            self.tables.initial_state_id,
            dtype=torch.long,
            device=self.device,
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        log_prob_sums = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        entropy_sums = torch.zeros_like(log_prob_sums)
        normalized_entropy_sums = torch.zeros_like(log_prob_sums)
        allowed_total = torch.zeros((), dtype=torch.float32, device=self.device)
        decision_total = torch.zeros((), dtype=torch.float32, device=self.device)
        pad_only = torch.zeros(vocab.size, dtype=torch.bool, device=self.device)
        pad_only[vocab.pad_id] = True

        for position in range(max_len + 1):
            active = ~done
            legal = self.tables.legal_actions[
                token_lengths.clamp_max(max_len), state_ids
            ]
            legal = torch.where(done.unsqueeze(1), pad_only.unsqueeze(0), legal)
            allowed_per_row = legal.sum(dim=1)
            allowed_total = allowed_total + torch.where(
                active, allowed_per_row, torch.zeros_like(allowed_per_row)
            ).sum()
            decision_total = decision_total + active.sum()

            if model is None:
                logits = torch.zeros(
                    (batch_size, vocab.size), dtype=torch.float32, device=self.device
                )
            else:
                logits, _ = model(sequences[:, : position + 1].clone())
            dist = Categorical(logits=logits.masked_fill(~legal, -torch.inf))
            action = dist.sample()
            log_prob_sums = log_prob_sums + dist.log_prob(action)
            entropy = dist.entropy()
            entropy_sums = entropy_sums + entropy
            normalizer = torch.log(allowed_per_row.clamp_min(2).to(entropy.dtype))
            normalized_entropy_sums = normalized_entropy_sums + torch.where(
                allowed_per_row > 1,
                entropy / normalizer,
                torch.zeros_like(entropy),
            )

            sequences[:, position + 1] = action
            is_formula = active & (action >= vocab.formula_offset)
            formula_rows = torch.where(is_formula)[0]
            formula_ids = action[formula_rows] - vocab.formula_offset
            previous_states = state_ids[formula_rows]
            token_ids[formula_rows, token_lengths[formula_rows]] = formula_ids
            instruction_codes = self.tables.emitted_codes[previous_states, formula_ids]
            emits = instruction_codes >= 0
            emit_rows = formula_rows[emits]
            vm_codes[emit_rows, vm_lengths[emit_rows]] = instruction_codes[emits]
            vm_lengths[emit_rows] += 1
            state_ids[formula_rows] = self.tables.next_states[previous_states, formula_ids]
            token_lengths[formula_rows] += 1
            done = done | (active & (action == vocab.eos_id))

        if not bool(done.all().item()):
            raise RuntimeError("Stage D tensor sampler left formulas unfinished")
        return FormulaTensorBatch(
            token_ids=token_ids,
            token_lengths=token_lengths,
            vm_codes=vm_codes,
            vm_lengths=vm_lengths,
            log_prob_sums=log_prob_sums,
            entropy_sums=entropy_sums,
            normalized_entropy_sums=normalized_entropy_sums,
            average_allowed_actions=allowed_total / decision_total,
        )