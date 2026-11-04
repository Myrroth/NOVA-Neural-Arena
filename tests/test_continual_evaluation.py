"""Synthetic pixels/tensors only: no real games or scientific checkpoints.

Every call supplies a fake factory. The default real factory is additionally
blocked, and no reserved scientific environment seed is used by these tests.
"""

import hashlib
import json
import random

import numpy as np
import pytest
import torch

from arena.continual.calibration import GateCalibration
from arena.continual import evaluation
from arena.continual.evaluation import TARGET_MODE, evaluate
from arena.continual.network import ContinualQNetwork
from arena.continual.policy import FrozenPolicyController
from arena.universal.evaluation import _local_rng as old_local_rng


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    def prohibited(*args, **kwargs):
        pytest.fail("A genuine simulator must never be created by these tests")

    monkeypatch.setattr(evaluation, "make_game", prohibited)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def good_calibration():
    count = 80
    h = torch.full((count, 8), 0.4)
    gate = GateCalibration()
    gate.refresh(
        h,
        h,
        torch.zeros_like(h),
        torch.zeros(count),
        torch.zeros(count),
        torch.full((count,), -20.0),
        torch.zeros(count, dtype=torch.bool),
        torch.arange(10).repeat_interleave(8),
        torch.ones(count, dtype=torch.bool),
        revision=0,
    )
    return gate


@pytest.fixture
def controller():
    # This random initialization is a tensor fixture, never a scientific model.
    with torch.random.fork_rng():
        torch.manual_seed(831171)
        net = (
            ContinualQNetwork(latent_size=8, hidden_size=8).requires_grad_(False).eval()
        )
    return FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0, context_length=2
    )


class FakePixels:
    identity = "synthetic-continual-v2-fixture"

    def __init__(self, game, *, max_steps, difficulty, scenario):
        self.maximum = min(max_steps, 4)
        self.steps = 0
        self.closed = False
        self.actions = []

    def _info(self):
        return {
            "environment_id": self.identity,
            "score": self.steps,
            "action_mask": np.array([True, True] + [False] * 8),
            "private_hint_for_logs_only": {"game_hint": 9},
        }

    def reset(self, *, seed):
        assert not 920001 <= seed <= 920020
        assert not 930001 <= seed <= 930020
        self.steps = 0
        self.actions = []
        return np.full((32, 32, 3), 10, dtype=np.uint8), self._info()

    def step(self, action):
        assert action in (0, 1)
        self.actions.append(action)
        self.steps += 1
        return (
            np.full((32, 32, 3), 10 + self.steps, dtype=np.uint8),
            3.0 if self.steps == 2 else -0.2,
            False,
            self.steps == self.maximum,
            self._info(),
        )

    def close(self):
        self.closed = True


def baseline():
    return {
        "snake": {
            "source": "training",
            "support": 32,
            "reward_mean": 0.1,
            "terminal_rate": 0.25,
        }
    }


def test_budget_hooks_count_interrupted_episode_before_logging(controller):
    calls = []
    physical = []

    def factory(*args, **kwargs):
        env = FakePixels(*args, **kwargs)
        physical.append(env)
        return env

    def account():
        calls.append("after")
        if calls.count("after") == 2:
            raise RuntimeError("synthetic resource STOP")

    with pytest.raises(RuntimeError, match="resource STOP"):
        run(
            controller,
            env_factory=factory,
            before_step=lambda: calls.append("before"),
            after_step=account,
        )
    assert calls == ["before", "after", "before", "after"]
    assert physical[0].steps == 2 and physical[0].closed


def run(controller, *, policy="shared", stage="after_adaptation", **kwargs):
    return evaluate(
        controller,
        831172 if controller is not None else 0,
        stage,
        policy,
        "snake",
        831173,
        env_factory=kwargs.pop("env_factory", FakePixels),
        max_steps=kwargs.pop("max_steps", 4),
        training_baselines=kwargs.pop("training_baselines", baseline()),
        **kwargs,
    )


def test_shared_after_pairs_use_same_actual_history_without_mutating_caller(
    controller, monkeypatch
):
    image = np.full((32, 32, 3), 9, dtype=np.uint8)
    mask = np.array([True, True] + [False] * 8)
    controller.act(image, 0, 0.0, mask)
    expected_controller = controller.fork()
    net = controller.network
    weights = {name: tensor.clone() for name, tensor in net.state_dict().items()}
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )
    calls = []
    original = net.step

    def observe(pixels, action, reward, hidden):
        assert pixels.shape == (1, 3, 32, 32) and pixels.dtype == torch.uint8
        assert action.shape == reward.shape == (1,)
        assert not torch.is_grad_enabled() and not net.training
        result = original(pixels, action, reward, hidden)
        calls.append(
            (
                pixels.clone(),
                action.clone(),
                reward.clone(),
                hidden.clone(),
                result[1].clone(),
            )
        )
        return result

    monkeypatch.setattr(net, "step", observe)
    row, pairs, predictions, trajectory = run(
        controller, record_trajectory=True, include_frames=True
    )
    assert row["steps"] == len(pairs) == len(predictions) == len(trajectory) == 4
    assert row["evaluation_schema"] == "continual-v2-evaluation-v1"
    assert row["return"] == pytest.approx(2.4)
    assert row["return_clipped"] == pytest.approx(0.4)
    assert row["private_hint_for_logs_only"] == '{"game_hint":9}'
    assert row["environment_id"] == FakePixels.identity
    # Actual observe, reset-memory counterfactual, hidden-pixel counterfactual,
    # continuous next-memory target. Counterfactuals add no environment steps.
    assert len(calls) == 16
    for index in range(4):
        actual, reset, hidden_pixels, target = calls[index * 4 : (index + 1) * 4]
        assert torch.count_nonzero(reset[3]) == 0
        assert torch.count_nonzero(hidden_pixels[0]) == 0
        assert torch.equal(actual[3], hidden_pixels[3])
        assert torch.equal(actual[4], target[3])
        if index % 2 == 0:
            assert torch.count_nonzero(actual[3]) == 0
        else:
            assert torch.equal(actual[3], calls[(index - 1) * 4][4])
        for alternative in (reset, hidden_pixels):
            assert torch.equal(actual[1], alternative[1])
            assert torch.equal(actual[2], alternative[2])
        expected_previous_reward = 0.0 if index == 0 else 1.0 if index == 2 else -0.2
        assert actual[2].item() == pytest.approx(expected_previous_reward)
        assert actual[1].item() == (
            0 if index == 0 else trajectory[index - 1]["action"]
        )
    assert [pair["planning_weight"] for pair in pairs] == [0.2, 0.0, 0.2, 0.0]
    assert [pair["planning_weight_gru_reset"] for pair in pairs] == [0.2, 0.0, 0.2, 0.0]
    assert [pair["planning_weight_pixels_hidden"] for pair in pairs] == [
        0.2,
        0.0,
        0.2,
        0.0,
    ]
    assert [pair["context_reset"] for pair in pairs] == [True, False, True, False]
    for pair in pairs:
        assert pair["action_world_off"] == int(
            np.nanargmax(
                [np.nan if q is None else q for q in json.loads(pair["q_direct"])]
            )
        )
        assert json.loads(pair["q_direct"])[2:] == [None] * 8
        assert pair["context_reset_gru_reset"]
        assert pair["context_reset_pixels_hidden"] == pair["context_reset"]
    assert (
        trajectory[0]["rgb_sha256"]
        == hashlib.sha256(trajectory[0]["rgb"].tobytes()).hexdigest()
    )
    assert not np.shares_memory(trajectory[0]["next_rgb"], trajectory[1]["rgb"])
    assert all(
        torch.equal(value, weights[name]) for name, value in net.state_dict().items()
    )
    assert all(not module.training for module in net.modules())
    assert all(parameter.grad is None for parameter in net.parameters())
    assert random.getstate() == rng[0]
    assert evaluation._numpy_state_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.random.get_rng_state(), rng[2])
    # The caller's pre-existing context did not advance or reset.
    actual = controller.act(image, 1, 0.2, mask)
    expected = expected_controller.act(image, 1, 0.2, mask)
    assert torch.equal(actual["hidden"], expected["hidden"])
    assert torch.equal(actual["q_mixed"], expected["q_mixed"])


def test_chosen_action_forecast_precedes_step_and_terminal_baseline_uses_training(
    controller, monkeypatch
):
    chosen = []
    observed_steps = [0]
    original = controller.network.model

    def record(hidden, actions):
        if len(actions) == 1:
            chosen.append((observed_steps[0], int(actions[0])))
        return original(hidden, actions)

    class OrderedFake(FakePixels):
        def step(self, action):
            assert chosen == [
                (i, previous) for i, previous in enumerate(self.actions)
            ] + [(self.steps, action)]
            result = super().step(action)
            observed_steps[0] = self.steps
            return result

    monkeypatch.setattr(controller.network, "model", record)
    _, _, predictions, _ = run(controller, env_factory=OrderedFake)
    assert [step for step, _ in chosen] == [0, 1, 2, 3]
    assert predictions[1]["reward_target_clipped"] == 1.0
    assert predictions[1]["reward_mae_baseline"] == pytest.approx(0.9)
    assert all(
        row["baseline_support"] == 32 and row["baseline_source"] == "training"
        for row in predictions
    )
    assert all(row["terminal_brier_baseline"] == 0.25**2 for row in predictions)
    assert predictions[-1]["truncated"] == 1 and predictions[-1]["terminated"] == 0


def test_world_target_is_continuation_even_when_next_control_will_reset(
    controller, monkeypatch
):
    calls = []

    def synthetic_step(pixels, action, reward, hidden):
        next_hidden = hidden + 0.25
        calls.append((hidden.clone(), next_hidden.clone()))
        return torch.tensor([[1.0, 0.0] + [99.0] * 8]), next_hidden, torch.zeros(1, 8)

    def synthetic_model(hidden, action):
        return hidden + 0.1, torch.zeros(len(action)), torch.zeros(len(action))

    def raw_values(hidden, mask, gamma):
        return torch.tensor([[0.0, 10.0] + [-torch.inf] * 8])

    monkeypatch.setattr(controller.network, "step", synthetic_step)
    monkeypatch.setattr(controller.network, "model", synthetic_model)
    monkeypatch.setattr(controller.network, "plan_values", raw_values)
    _, pairs, predictions, _ = run(controller)
    assert all(row["target_mode"] == TARGET_MODE for row in predictions)
    assert [row["next_context_boundary"] for row in predictions] == [
        False,
        True,
        False,
        True,
    ]
    assert [row["latent_mse_model"] for row in predictions] == pytest.approx(
        [0.15**2] * 4
    )
    assert [row["latent_mse_copy"] for row in predictions] == pytest.approx(
        [0.25**2] * 4
    )
    # Boundary target observes the next image from .5, reaching .75; the next
    # actual control observation starts at zero and reaches .25 instead.
    assert torch.all(calls[7][0] == 0.5) and torch.all(calls[7][1] == 0.75)
    assert torch.all(calls[8][0] == 0) and torch.all(calls[8][1] == 0.25)
    assert pairs[0]["action_default"] == 1 and pairs[1]["action_default"] == 0
    assert pairs[1]["model_values"] != pairs[1]["q_mixed"]
    assert pairs[1]["q_mixed"] == pairs[1]["q_direct"]


@pytest.mark.parametrize(
    "policy", ["scratch", "untrained", "world_off", "gru_reset", "pixels_hidden"]
)
def test_standalone_interventions_have_own_history_and_only_declared_inputs(
    controller, policy, monkeypatch
):
    calls = []
    original = controller.network.step

    def record(pixels, action, reward, hidden):
        output = original(pixels, action, reward, hidden)
        calls.append(
            (
                pixels.clone(),
                action.clone(),
                reward.clone(),
                hidden.clone(),
                output[1].clone(),
            )
        )
        return output

    monkeypatch.setattr(controller.network, "step", record)
    row, pairs, predictions, trajectory = run(
        controller,
        policy=policy,
        stage="ablation",
        record_world_predictions=False,
        record_trajectory=True,
    )
    assert row["policy"] == policy
    assert pairs == predictions == []
    assert len(calls) == len(trajectory) == 4
    for index, call in enumerate(calls):
        if policy == "pixels_hidden":
            assert torch.count_nonzero(call[0]) == 0
        else:
            assert torch.count_nonzero(call[0]) > 0
        if index % 2 == 0 or policy == "gru_reset":
            assert torch.count_nonzero(call[3]) == 0
        else:
            assert torch.equal(call[3], calls[index - 1][4])
        assert call[1].item() == (0 if index == 0 else trajectory[index - 1]["action"])
    if policy == "world_off":
        assert all(
            frame["planning_weight"] == 0.0 and frame["q_mixed"] == frame["q_direct"]
            for frame in trajectory
        )
    if policy == "gru_reset":
        assert all(frame["context_reset"] for frame in trajectory)
        assert trajectory[0]["planning_weight"] == 0.2
        assert trajectory[1]["planning_weight"] == 0.0


def test_a_cold_gate_stays_exactly_direct_even_with_large_raw_values(
    controller, monkeypatch
):
    actor = FrozenPolicyController(
        controller.network, calibration=GateCalibration(), policy_revision=0
    )
    monkeypatch.setattr(
        actor.network,
        "plan_values",
        lambda hidden, mask, gamma: torch.tensor(
            [[1.0e12, -1.0e12] + [-torch.inf] * 8]
        ),
    )
    _, pairs, predictions, trajectory = run(actor, record_trajectory=True)
    assert predictions
    for pair, frame in zip(pairs, trajectory):
        assert pair["planning_weight"] == 0.0
        assert pair["model_values"] != pair["q_mixed"]
        assert pair["q_mixed"] == pair["q_direct"] == frame["q_mixed"]


def test_random_is_repeatable_independent_and_uses_the_new_rng_namespace():
    first = run(None, policy="random", stage="baseline", record_trajectory=True)
    second = run(None, policy="random", stage="baseline", record_trajectory=True)
    assert first == second
    assert first[1] == first[2] == []
    assert first[0]["network_sha256"] == "none-random"
    assert all(frame["q_mixed"] is None for frame in first[3])
    identity = (0, "snake", 831173, "baseline", "random")
    new = evaluation._local_rng(*identity).integers(0, 2**32, 16)
    old = old_local_rng(*identity).integers(0, 2**32, 16)
    assert not np.array_equal(new, old)


def test_demonstration_frames_preserve_observed_pixels_when_environment_reuses_buffer(
    controller,
):
    class BufferedFake(FakePixels):
        def reset(self, *, seed):
            observation, info = super().reset(seed=seed)
            self.buffer = observation
            return self.buffer, info

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            self.buffer[:] = observation
            return self.buffer, reward, terminated, truncated, info

    _, _, _, trajectory = run(
        controller,
        env_factory=BufferedFake,
        record_trajectory=True,
        include_frames=True,
    )
    for step, frame in enumerate(trajectory):
        assert np.all(frame["rgb"] == 10 + step)
        assert np.all(frame["next_rgb"] == 11 + step)
        assert frame["rgb_sha256"] != frame["next_rgb_sha256"]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {
            "snake": {
                "source": "evaluation",
                "support": 32,
                "reward_mean": 0.1,
                "terminal_rate": 0.25,
            }
        },
    ],
)
def test_missing_or_test_derived_baselines_fail_before_any_environment(controller, bad):
    def prohibited(*args, **kwargs):
        pytest.fail(
            "invalid baseline provenance must fail before environment construction"
        )

    with pytest.raises(ValueError, match="baselines|provenance"):
        run(controller, training_baselines=bad, env_factory=prohibited)


def test_changed_mechanical_mask_is_rejected_and_environment_closed(controller):
    instances = []

    class BadMask(FakePixels):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def step(self, action):
            result = list(super().step(action))
            result[4]["action_mask"][1] = False
            return tuple(result)

    with pytest.raises(ValueError, match="mask changed"):
        run(controller, env_factory=BadMask)
    assert instances[0].closed


def test_global_rng_contamination_is_restored_and_rejected(controller):
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )

    class DirtyFake(FakePixels):
        def step(self, action):
            random.random()
            np.random.random()
            torch.rand(1)
            return super().step(action)

    with pytest.raises(RuntimeError, match="global RNG"):
        run(controller, env_factory=DirtyFake)
    assert random.getstate() == before[0]
    assert evaluation._numpy_state_equal(np.random.get_state(), before[1])
    assert torch.equal(torch.random.get_rng_state(), before[2])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_steps": 401}, "1..400"),
        ({"include_frames": True}, "frames require"),
        ({"record_trajectory": 1}, "boolean"),
        ({"policy": "world_off", "record_world_predictions": True}, "shared-after"),
        ({"difficulty": float("nan")}, "difficulty"),
        ({"scenario": "threat"}, "Agar-like"),
    ],
)
def test_bad_controls_fail_before_environment(controller, kwargs, match):
    def prohibited(*args, **options):
        pytest.fail("invalid controls must not construct an environment")

    with pytest.raises(ValueError, match=match):
        run(controller, env_factory=prohibited, **kwargs)


def test_random_refuses_a_neural_controller(controller):
    with pytest.raises(ValueError, match="model_seed=0"):
        run(controller, policy="random", stage="baseline")
