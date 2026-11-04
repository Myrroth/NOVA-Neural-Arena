"""Analytical tensor fixtures only: no games, policies, replay or campaign."""

import math

import pytest

torch = pytest.importorskip("torch")

from arena.continual.objectives import (
    masked_sequence_mean,
    n_step_targets,
    policy_distillation,
)


def test_sequence_mean_does_not_turn_eight_steps_into_eight_votes():
    values = torch.tensor([[2.0] * 8, [10.0] + [999.0] * 7], requires_grad=True)
    valid = torch.tensor([[True] * 8, [True] + [False] * 7])
    result = masked_sequence_mean(values, valid)
    assert result.item() == pytest.approx(6.0)
    result.backward()
    torch.testing.assert_close(values.grad[0], torch.full((8,), 1 / 16))
    assert values.grad[1, 0].item() == pytest.approx(0.5)
    assert torch.count_nonzero(values.grad[1, 1:]) == 0


def test_weighted_sequence_mean_excludes_wholly_padded_rows():
    values = torch.tensor(
        [[2.0, 2.0], [10.0, 999.0], [999.0, 999.0]], dtype=torch.float64
    )
    valid = torch.tensor([[True, True], [True, False], [False, False]])
    result = masked_sequence_mean(values, valid, torch.tensor([3.0, 1.0, 1000.0]))
    assert result.item() == pytest.approx(4.0)
    # The normalization stays finite even if summing raw weights would overflow.
    assert masked_sequence_mean(
        values.float(), valid, torch.tensor([1e38, 1e38, 1e38])
    ).item() == pytest.approx(6.0)


@pytest.mark.parametrize("all_padding", [True, False])
def test_empty_effective_weight_is_differentiable_zero(all_padding):
    values = torch.tensor([[3.0, 4.0]], requires_grad=True)
    valid = torch.full((1, 2), not all_padding, dtype=torch.bool)
    result = masked_sequence_mean(values, valid, torch.zeros(1))
    assert result.item() == 0.0
    result.backward()
    assert torch.count_nonzero(values.grad) == 0


def test_distillation_matches_manual_legal_kl_with_temperature_squared():
    # Illegal action has an enormous logit in both networks; it cannot enter
    # normalization or pull either legal action's gradient.
    teacher = torch.tensor(
        [[[math.log(3.0), 0.0, 1000.0]]], dtype=torch.float64, requires_grad=True
    )
    student = torch.tensor(
        [[[0.0, 0.0, -1000.0]]], dtype=torch.float64, requires_grad=True
    )
    mask = torch.tensor([[[True, True, False]]])
    valid = torch.ones((1, 1), dtype=torch.bool)
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    loss = policy_distillation(student, teacher, mask, valid)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(
        student.grad, torch.tensor([[[-0.25, 0.25, 0.0]]], dtype=torch.float64)
    )
    temperature_loss = policy_distillation(
        student.detach(), teacher.detach() * 2, mask, valid, temp=2.0
    )
    assert temperature_loss.item() == pytest.approx(expected * 4)


def test_distillation_balances_sequences_and_ignores_padding_and_illegal_logits():
    student = torch.zeros((3, 8, 3), requires_grad=True)
    teacher = torch.zeros_like(student)
    teacher[0, :, 0] = math.log(3.0)
    teacher[1, 0, 0] = math.log(9.0)
    teacher[..., 2] = 1000.0
    masks = torch.zeros_like(student, dtype=torch.bool)
    masks[:2, :, :2] = True
    valid = torch.zeros((3, 8), dtype=torch.bool)
    valid[0] = True
    valid[1, 0] = True
    expected_a = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    expected_b = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
    loss = policy_distillation(student, teacher, masks, valid)
    assert loss.item() == pytest.approx((expected_a + expected_b) / 2, abs=1e-7)
    loss.backward()
    assert torch.count_nonzero(student.grad[~valid]) == 0
    assert torch.count_nonzero(student.grad[..., 2]) == 0


def test_distillation_single_legal_action_and_all_padding_have_zero_gradient():
    student = torch.tensor([[[10.0, -3.0], [2.0, 5.0]]], requires_grad=True)
    teacher = torch.tensor([[[-10.0, 3.0], [5.0, -2.0]]], requires_grad=True)
    masks = torch.tensor([[[True, False], [False, False]]])
    valid = torch.tensor([[True, False]])
    loss = policy_distillation(student, teacher, masks, valid)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None


def test_n_step_exact_three_rewards_and_discounted_successor():
    rewards = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True
    )
    bootstrap = torch.tensor(
        [[999.0, 10.0, 20.0, 30.0, 40.0]], dtype=torch.float64, requires_grad=True
    )
    flags = torch.zeros_like(rewards, dtype=torch.bool)
    result = n_step_targets(rewards, flags, flags, ~flags, bootstrap, 0.5, 3)
    # Horizon n=3, then shorter horizons at the end of available data.
    expected = torch.tensor(
        [
            [
                1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 30,
                2 + 0.5 * 3 + 0.25 * 4 + 0.125 * 40,
                3 + 0.5 * 4 + 0.25 * 40,
                4 + 0.5 * 40,
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)
    assert not result.requires_grad and result.grad_fn is None
    assert rewards.grad is bootstrap.grad is None


def test_terminal_cuts_but_truncation_bootstraps_real_final_successor():
    rewards = torch.tensor([[1.0, 2.0, 999.0], [1.0, 2.0, 999.0]])
    terminal = torch.tensor([[False, True, False], [False, False, False]])
    truncated = torch.tensor([[False, False, False], [False, True, False]])
    valid = torch.ones_like(terminal)
    bootstrap = torch.tensor([[0.0, 10.0, 20.0, 9999.0], [0.0, 10.0, 20.0, 9999.0]])
    result = n_step_targets(rewards, terminal, truncated, valid, bootstrap, 0.5, 3)
    # The later valid reward is a sentinel for a subsequent episode. Neither
    # preceding return may include it after its terminal/truncation boundary.
    assert result[0, :2].tolist() == pytest.approx([2.0, 2.0])
    assert result[1, :2].tolist() == pytest.approx([7.0, 12.0])


def test_padding_bootstraps_last_real_successor_and_never_crosses_a_gap():
    rewards = torch.tensor([[1.0, 2.0, 777.0, 888.0], [1.0, 777.0, 9.0, 888.0]])
    valid = torch.tensor([[True, True, False, False], [True, False, True, False]])
    flags = torch.zeros_like(valid)
    bootstrap = torch.tensor(
        [[0.0, 10.0, 20.0, 333.0, 444.0], [0.0, 10.0, 222.0, 30.0, 444.0]]
    )
    result = n_step_targets(rewards, flags, flags, valid, bootstrap, 0.5, 3)
    torch.testing.assert_close(
        result, torch.tensor([[7.0, 12.0, 0.0, 0.0], [6.0, 0.0, 24.0, 0.0]])
    )


def test_both_terminal_flags_use_true_termination_and_zero_gamma_uses_reward():
    rewards = torch.tensor([[1.0, 2.0]])
    both = torch.tensor([[True, False]])
    valid = torch.ones_like(both)
    values = torch.tensor([[100.0, 200.0, 300.0]])
    result = n_step_targets(rewards, both, both, valid, values, 1.0, 100)
    torch.testing.assert_close(result, torch.tensor([[1.0, 302.0]]))
    torch.testing.assert_close(
        n_step_targets(rewards, both, both, valid, values, 0.0, 3), rewards
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_are_rejected_even_under_padding(bad):
    invalid = torch.tensor([[bad]])
    valid = torch.zeros((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="finite"):
        masked_sequence_mean(invalid, valid)
    with pytest.raises(ValueError, match="finite"):
        masked_sequence_mean(torch.ones((1, 1)), valid, torch.tensor([bad]))
    with pytest.raises(ValueError, match="finite"):
        policy_distillation(
            invalid.unsqueeze(-1), torch.zeros((1, 1, 1)), valid.unsqueeze(-1), valid
        )
    with pytest.raises(ValueError, match="finite"):
        n_step_targets(invalid, valid, valid, valid, torch.zeros((1, 2)), 0.9, 3)
    with pytest.raises(ValueError, match="finite"):
        n_step_targets(
            torch.zeros((1, 1)), valid, valid, valid, torch.tensor([[0.0, bad]]), 0.9, 3
        )


def test_shape_boolean_and_nonnegative_weight_contracts_fail_early():
    values = torch.ones((2, 3))
    valid = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="bool"):
        masked_sequence_mean(values, valid.float())
    with pytest.raises(ValueError, match="nonnegative"):
        masked_sequence_mean(values, valid, torch.tensor([1.0, -1.0]))
    with pytest.raises(ValueError, match="batch"):
        masked_sequence_mean(values, valid, torch.ones(3))
    logits = values.unsqueeze(-1).expand(-1, -1, 2)
    with pytest.raises(ValueError, match="legal action"):
        policy_distillation(
            logits, logits, torch.zeros_like(logits, dtype=torch.bool), valid
        )
    with pytest.raises(ValueError, match="teacher_q"):
        policy_distillation(
            logits, logits[:, :, :1], torch.ones_like(logits, dtype=torch.bool), valid
        )
    with pytest.raises(ValueError, match="time\\+1"):
        n_step_targets(values, valid, valid, valid, values, 0.9, 3)
    with pytest.raises(ValueError, match="bool"):
        n_step_targets(values, valid.float(), valid, valid, torch.ones((2, 4)), 0.9, 3)


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), True])
def test_invalid_discount_is_rejected(gamma):
    flags = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="gamma"):
        n_step_targets(
            torch.ones((1, 1)), flags, flags, flags, torch.ones((1, 2)), gamma, 3
        )


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_n_step_is_rejected(n):
    flags = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="n_step"):
        n_step_targets(
            torch.ones((1, 1)), flags, flags, flags, torch.ones((1, 2)), 0.9, n
        )


@pytest.mark.parametrize("temp", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_temperature_is_rejected(temp):
    logits = torch.ones((1, 1, 2))
    with pytest.raises(ValueError, match="temp"):
        policy_distillation(
            logits,
            logits,
            torch.ones_like(logits, dtype=torch.bool),
            torch.ones((1, 1), dtype=torch.bool),
            temp=temp,
        )


@pytest.mark.parametrize("temp", [1e308, 1e-308])
def test_unrepresentable_temperature_fails_with_controlled_finite_value_error(temp):
    student = torch.tensor([[[1.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 1.0]]])
    with pytest.raises(ValueError, match="finite"):
        policy_distillation(
            student,
            teacher,
            torch.ones_like(student, dtype=torch.bool),
            torch.ones((1, 1), dtype=torch.bool),
            temp=temp,
        )
