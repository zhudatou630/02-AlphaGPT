from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.factors import FACTOR_NAMES, build_factor_values_numpy
from alpha_etf.research_v3a.language import (
    FORMULA_VOCAB,
    MAX_FORMULA_TOKENS,
    TOKEN_NAMES,
    ExpressionBuilder,
    GrammarState,
    allowed_formula_token_ids,
    compile_formula,
    minimum_tokens_to_finish,
    parse_formula,
    transition_state,
)
from alpha_etf.research_v3a.sampling import (
    PolicyVocab,
    SamplingConfig,
    allowed_policy_action_ids,
    build_action_mask,
    generate_random_formula,
    sample_formulas,
)
from alpha_etf.research_v3a.torch_vm import BatchTorchVM, compiled_to_tensor
from alpha_etf.research_v3a.vm import StackVM


def _factor_panel() -> tuple[np.ndarray, np.ndarray]:
    dates = 90
    close = 100.0 + np.arange(dates, dtype=float)
    absolute = np.stack([close - 0.5, close + 1.0, close - 1.0, close])[None, ...]
    mask = np.ones((1, dates), dtype=bool)
    return build_factor_values_numpy(absolute, mask), mask


class V3ALanguageTests(unittest.TestCase):
    def test_vocab_ids_are_frozen(self) -> None:
        self.assertEqual(FORMULA_VOCAB.size, 27)
        self.assertEqual(FORMULA_VOCAB.token_names, TOKEN_NAMES)
        self.assertEqual(FORMULA_VOCAB.name_to_id("DAYRET"), 0)
        self.assertEqual(FORMULA_VOCAB.name_to_id("WIN_1"), 11)
        self.assertEqual(FORMULA_VOCAB.name_to_id("ADD"), 17)
        self.assertEqual(FORMULA_VOCAB.name_to_id("CONST_1"), 26)
        self.assertEqual(PolicyVocab().size, 30)

    def test_builder_parses_parameter_atoms_and_rpn(self) -> None:
        names = ["DAYRET", "ROC", "WIN_20", "ADD", "WIN_5", "REF"]
        expression = parse_formula(FORMULA_VOCAB.encode(names))
        self.assertEqual(expression.text(), "REF(ADD(DAYRET,ROC(20)),5)")
        compiled = compile_formula(FORMULA_VOCAB.encode(names))
        self.assertLessEqual(len(compiled.instructions), len(names))
        self.assertEqual(compiled.expression.to_dict(), expression.to_dict())

    def test_invalid_parameter_sequences_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_formula(FORMULA_VOCAB.encode(["ROC", "WIN_1"]))
        with self.assertRaises(ValueError):
            parse_formula(FORMULA_VOCAB.encode(["MA_RATIO", "WIN_20", "WIN_10"]))
        with self.assertRaises(ValueError):
            parse_formula(FORMULA_VOCAB.encode(["DAYRET", "WIN_1", "MEAN"]))
        with self.assertRaises(ValueError):
            parse_formula(FORMULA_VOCAB.encode(["ADD"]))

    def test_action_mask_tracks_pending_windows_and_eos(self) -> None:
        policy = PolicyVocab()
        state = GrammarState()
        initial = set(
            allowed_formula_token_ids(state, formula_length=0, max_length=MAX_FORMULA_TOKENS)
        )
        self.assertIn(FORMULA_VOCAB.name_to_id("DAYRET"), initial)
        self.assertIn(FORMULA_VOCAB.name_to_id("ROC"), initial)
        self.assertNotIn(FORMULA_VOCAB.name_to_id("WIN_5"), initial)
        state = transition_state(state, FORMULA_VOCAB.id_to_token(FORMULA_VOCAB.name_to_id("ROC")))
        allowed = set(allowed_formula_token_ids(state, formula_length=1))
        self.assertEqual(
            allowed,
            {FORMULA_VOCAB.name_to_id(f"WIN_{window}") for window in (5, 10, 20, 40, 60)},
        )
        state = transition_state(
            GrammarState(), FORMULA_VOCAB.id_to_token(FORMULA_VOCAB.name_to_id("DAYRET"))
        )
        mask = build_action_mask(
            [state],
            torch.tensor([1]),
            torch.tensor([False]),
            policy,
            SamplingConfig(),
        )
        self.assertEqual(mask[0, policy.eos_id].item(), 0.0)
        self.assertEqual(
            mask[0, policy.to_model_id(FORMULA_VOCAB.name_to_id("WIN_1"))].item(), 0.0
        )

    def test_random_generator_produces_only_valid_formulas(self) -> None:
        rng = random.Random(123)
        lengths = []
        for _ in range(10_000):
            formula = generate_random_formula(rng)
            self.assertGreaterEqual(len(formula), 1)
            self.assertLessEqual(len(formula), MAX_FORMULA_TOKENS)
            parse_formula(formula)
            lengths.append(len(formula))
        self.assertGreater(len(set(lengths)), 5)

    def test_all_reachable_grammar_states_can_close_within_token_budget(self) -> None:
        policy = PolicyVocab()
        config = SamplingConfig()
        frontier = {(GrammarState(), 0)}
        visited = set()
        saw_pending_ratio = False
        saw_window_one = False
        saw_length_15 = False
        while frontier:
            state, length = frontier.pop()
            if (state, length) in visited:
                continue
            visited.add((state, length))
            actions = allowed_policy_action_ids(
                state,
                formula_length=length,
                policy_vocab=policy,
                config=config,
            )
            self.assertTrue(actions)
            self.assertEqual(policy.eos_id in actions, state.valid)
            if length == MAX_FORMULA_TOKENS:
                self.assertEqual(actions, (policy.eos_id,))
                saw_length_15 = True
                continue
            for model_id in actions:
                if model_id == policy.eos_id:
                    continue
                token_id = policy.to_formula_id(model_id)
                token = FORMULA_VOCAB.id_to_token(token_id)
                next_state = transition_state(state, token)
                remaining = MAX_FORMULA_TOKENS - length - 1
                self.assertLessEqual(minimum_tokens_to_finish(next_state), remaining)
                saw_pending_ratio |= next_state.pending_factor == "MA_RATIO"
                saw_window_one |= bool(
                    next_state.stack and next_state.stack[-1] == "window:1"
                )
                frontier.add((next_state, length + 1))
        self.assertTrue(saw_pending_ratio)
        self.assertTrue(saw_window_one)
        self.assertTrue(saw_length_15)

    def test_transformer_sampler_outputs_valid_formulas(self) -> None:
        torch.manual_seed(7)
        policy_vocab = PolicyVocab()
        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=policy_vocab.size,
                max_sequence_len=MAX_FORMULA_TOKENS + 2,
                d_model=16,
                num_layers=1,
                num_heads=4,
                ff_dim=32,
                dropout=0.0,
            )
        )
        batch = sample_formulas(
            model, 16, policy_vocab, SamplingConfig(), torch.device("cpu")
        )
        self.assertEqual(batch.log_prob_sums.shape, (16,))
        for formula in batch.formulas:
            parse_formula(formula)

    def test_cpu_and_torch_vm_match_compiled_formulas(self) -> None:
        factor_values, mask = _factor_panel()
        formulas = [
            FORMULA_VOCAB.encode(["DAYRET"]),
            FORMULA_VOCAB.encode(["ROC", "WIN_20"]),
            FORMULA_VOCAB.encode(["DAYRET", "ROC", "WIN_20", "ADD"]),
            FORMULA_VOCAB.encode(["PRICE_MA", "WIN_20", "WIN_5", "REF"]),
            FORMULA_VOCAB.encode(["DAYRET", "WIN_10", "MEAN", "NEG", "ABS"]),
            FORMULA_VOCAB.encode(["MA_RATIO", "WIN_5", "WIN_20", "SIGN"]),
        ]
        compiled = [compile_formula(formula) for formula in formulas]
        cpu = [StackVM().execute_compiled(item, factor_values, mask) for item in compiled]
        codes, lengths = compiled_to_tensor(compiled, device=torch.device("cpu"))
        torch_result = BatchTorchVM().execute(
            codes,
            lengths,
            torch.as_tensor(factor_values, dtype=torch.float32),
            torch.as_tensor(mask),
        )
        self.assertTrue(bool(torch_result.valid.all().item()))
        for index, result in enumerate(cpu):
            self.assertTrue(result.valid, result.invalid_reason)
            np.testing.assert_allclose(
                result.signal,
                torch_result.signal[index].numpy(),
                rtol=2e-5,
                atol=2e-6,
                equal_nan=True,
            )

    def test_torch_vm_memory_plan_chunks_training_shape(self) -> None:
        formula = compile_formula(
            FORMULA_VOCAB.encode(
                ["DAYRET", "ROC", "WIN_20", "ADD", "CLV", "MUL"]
            )
        )
        formulas = [formula] * 4096
        codes, lengths = compiled_to_tensor(formulas, device=torch.device("cpu"))
        factor_values = torch.empty((40, 35, 1400), dtype=torch.float32)
        mask = torch.ones((35, 1400), dtype=torch.bool)
        vm = BatchTorchVM(max_working_bytes=512 * 1024**2)
        plan = vm.execution_plan(codes, lengths, factor_values, mask)
        self.assertLess(plan.chunk_size, 4096)
        self.assertLessEqual(plan.estimated_working_bytes, 512 * 1024**2)
        self.assertLess(plan.output_bytes, 2 * 1024**3)
        self.assertLessEqual(plan.estimated_peak_bytes, 3 * 1024**3)

        full_factor_values = torch.empty((40, 35, 5196), dtype=torch.float32)
        full_mask = torch.ones((35, 5196), dtype=torch.bool)
        with self.assertRaises(MemoryError):
            vm.execution_plan(codes, lengths, full_factor_values, full_mask)

    def test_forced_formula_chunking_preserves_vm_results(self) -> None:
        factor_values, mask = _factor_panel()
        formulas = [
            compile_formula(FORMULA_VOCAB.encode(["DAYRET"])),
            compile_formula(FORMULA_VOCAB.encode(["ROC", "WIN_20"])),
            compile_formula(FORMULA_VOCAB.encode(["DAYRET", "GAP", "ADD"])),
        ]
        codes, lengths = compiled_to_tensor(formulas, device=torch.device("cpu"))
        factors = torch.as_tensor(factor_values, dtype=torch.float32)
        torch_mask = torch.as_tensor(mask)
        normal = BatchTorchVM().execute(codes, lengths, factors, torch_mask)
        bytes_per_panel = mask.size * factors.element_size()
        chunked = BatchTorchVM(
            max_working_bytes=5 * bytes_per_panel,
            max_output_bytes=100 * bytes_per_panel,
            max_total_bytes=105 * bytes_per_panel,
        ).execute(codes, lengths, factors, torch_mask)
        self.assertTrue(torch.equal(normal.valid, chunked.valid))
        self.assertTrue(torch.equal(normal.invalid_code, chunked.invalid_code))
        np.testing.assert_allclose(
            normal.signal.numpy(), chunked.signal.numpy(), rtol=0.0, atol=0.0, equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()