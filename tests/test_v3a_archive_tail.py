from __future__ import annotations

import unittest

import numpy as np
import torch

from alpha_etf.gpt.policy import TransformerFormulaPolicy, TransformerPolicyConfig
from alpha_etf.research_v3a.archive_tail import (
    ArchiveTailConfig,
    ArchiveTailState,
    archive_tail_objective,
)
from alpha_etf.research_v3a.attempts import ATTEMPT_DTYPE, AttemptStatus
from alpha_etf.research_v3a.gpu_sampling import TensorFormulaSampler
from alpha_etf.research_v3a.language import MAX_FORMULA_TOKENS
from alpha_etf.research_v3a.sampling import PolicyVocab


def _records(rows: list[tuple[int, int, float, int]]) -> np.ndarray:
    records = np.zeros(len(rows), dtype=ATTEMPT_DTYPE)
    records["token_ids"] = np.uint8(255)
    records["training_reward"] = np.float32(0.0)
    for index, (attempt, hash_byte, reward, status) in enumerate(rows):
        records["attempt_index"][index] = attempt
        records["training_step"][index] = 1
        records["status"][index] = status
        records["token_len"][index] = 1
        records["token_ids"][index, 0] = hash_byte
        records["canonical_hash"][index] = hash_byte
        records["reward"][index] = reward
    return records


class ArchiveTailTests(unittest.TestCase):
    def test_state_classifies_and_weights_three_groups(self) -> None:
        valid = int(AttemptStatus.PENDING_SEMANTIC_VALID)
        invalid = int(AttemptStatus.VM_INVALID)
        state = ArchiveTailState(
            ArchiveTailConfig(elite_fraction=0.5, archive_size=2)
        )
        first = _records(
            [
                (0, 1, 1.0, valid),
                (1, 1, 1.0, valid),
                (2, 2, 2.0, valid),
                (3, 3, np.nan, invalid),
            ]
        )
        labels = state.apply(first, prepared=True)
        self.assertEqual(labels.metrics["new_valid_count"], 2)
        self.assertEqual(labels.metrics["elite_count"], 1)
        self.assertEqual(labels.metrics["within_batch_duplicate_count"], 1)
        self.assertEqual(labels.metrics["semantic_invalid_count"], 1)
        np.testing.assert_allclose(labels.elite_weights, [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(labels.archive_weights, 0.0)
        np.testing.assert_allclose(labels.waste_weights, [0.0, 0.5, 0.0, 0.5])
        np.testing.assert_allclose(labels.combined_weights, [0.0, -0.5, 1.0, -0.5])
        self.assertEqual(len(state.seen), 3)
        self.assertEqual(len(state.archive), 2)

        second = _records(
            [
                (4, 1, 1.0, valid),
                (5, 4, 3.0, valid),
                (6, 5, 0.5, valid),
                (7, 5, 0.5, valid),
            ]
        )
        labels = state.apply(second, prepared=True)
        self.assertEqual(labels.metrics["history_duplicate_count"], 1)
        self.assertEqual(labels.metrics["within_batch_duplicate_count"], 1)
        self.assertEqual(labels.metrics["archive_improver_count"], 1)
        np.testing.assert_allclose(labels.elite_weights, [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(labels.archive_weights, [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(labels.waste_weights, [0.5, 0.0, 0.0, 0.5])
        np.testing.assert_allclose(labels.combined_weights, [-0.5, 2.0, 0.0, -0.5])
        self.assertEqual({entry.reward for entry in state.archive.values()}, {2.0, 3.0})

        log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
        objective = archive_tail_objective(
            log_prob_sums=log_probs,
            elite_weights=torch.as_tensor(labels.elite_weights),
            archive_weights=torch.as_tensor(labels.archive_weights),
            waste_weights=torch.as_tensor(labels.waste_weights),
        )
        self.assertEqual(float(objective.elite_loss.detach()), 2.0)
        self.assertEqual(float(objective.archive_loss.detach()), 2.0)
        self.assertEqual(float(objective.waste_loss.detach()), -2.5)
        self.assertEqual(float(objective.loss.detach()), 1.5)
        objective.loss.backward()
        np.testing.assert_allclose(log_probs.grad.numpy(), [0.5, -2.0, 0.0, 0.5])

        third = _records(
            [
                (8, 6, 4.0, valid),
                (9, 7, 5.0, valid),
            ]
        )
        labels = state.apply(third, prepared=True)
        np.testing.assert_allclose(labels.archive_weights, [0.4, 0.6])
        self.assertEqual(labels.metrics["archive_improver_count"], 2)

        zero = torch.zeros(2)
        empty = archive_tail_objective(
            log_prob_sums=torch.tensor([-1.0, -2.0], requires_grad=True),
            elite_weights=zero,
            archive_weights=zero,
            waste_weights=zero,
        )
        self.assertEqual(float(empty.loss.detach()), 0.0)

    def test_valid_representative_wins_and_random_does_not_pollute_seen(self) -> None:
        valid = int(AttemptStatus.PENDING_SEMANTIC_VALID)
        invalid = int(AttemptStatus.VM_INVALID)
        state = ArchiveTailState(ArchiveTailConfig(elite_fraction=1.0, archive_size=2))
        random_only = _records([(0, 9, 9.0, valid)])
        self.assertNotIn(random_only[0]["canonical_hash"].tobytes(), state.seen)
        model = _records(
            [
                (1, 9, np.nan, invalid),
                (2, 9, 1.0, valid),
            ]
        )
        labels = state.apply(model, prepared=True)
        self.assertEqual(labels.new_valid_indices, (1,))
        self.assertEqual(labels.new_canonical_indices, (1,))
        np.testing.assert_allclose(labels.combined_weights, [-1.0, 1.0])

    def test_teacher_forcing_matches_sequential_policy_log_prob(self) -> None:
        device = torch.device("cpu")
        vocab = PolicyVocab()
        model = TransformerFormulaPolicy(
            TransformerPolicyConfig(
                model_vocab_size=vocab.size,
                max_sequence_len=MAX_FORMULA_TOKENS + 1,
                d_model=8,
                num_layers=1,
                num_heads=2,
                ff_dim=16,
                dropout=0.0,
            )
        )
        sampler = TensorFormulaSampler(device=device, policy_vocab=vocab)
        torch.manual_seed(91)
        sampled = sampler.sample_policy(model, 12)
        recomputed = sampler.score_policy_sequences(
            model,
            sampled.token_ids.detach(),
            sampled.token_lengths.detach(),
        )
        torch.testing.assert_close(
            recomputed.detach(), sampled.log_prob_sums.detach(), rtol=1e-6, atol=1e-6
        )
        sequence = torch.full((3, 7), vocab.bos_id, dtype=torch.long)
        all_logits = model.forward_all(sequence)
        last_logits, _ = model(sequence)
        torch.testing.assert_close(all_logits[:, -1], last_logits)
        (-recomputed.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()