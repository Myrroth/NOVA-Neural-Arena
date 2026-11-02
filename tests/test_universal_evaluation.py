"""Inference fixtures and <=25 real random-initialized technical steps only.

No reserved 920001..920020 seeds, optimizers, training or campaign is used.
Scripted tensor values in unit tests are not player demonstrations or replay.
"""
import copy
import json
import random

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from arena.universal.evaluation import _decide, evaluate
from arena.universal.network import VisualQNetwork


@pytest.fixture(autouse=True)
def limited_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def network():
    # Initialization is a fixture outside evaluation, never a trained checkpoint.
    with torch.random.fork_rng():
        torch.manual_seed(104)
        return VisualQNetwork()


class ToyPixels:
    """A four-transition instrumentation fixture, not a genuine game result."""
    identity = "synthetic-visual-fixture-v1"

    def __init__(self, game, *, max_steps, difficulty, scenario):
        self.maximum = min(max_steps, 4)
        self.steps = 0
        self.closed = False

    def _info(self):
        return {"environment_id": self.identity, "action_mask": np.array([True, True]+[False]*8),
                "score": self.steps, "private_hint_must_never_reach_policy": [9, 8, 7],
                "terminal_reason": "fixture_limit" if self.steps == self.maximum else ""}

    def reset(self, *, seed):
        assert not 920001 <= seed <= 920020
        self.steps = 0
        return np.full((32, 32, 3), 10, dtype=np.uint8), self._info()

    def step(self, action):
        assert action in (0, 1)
        self.steps += 1
        return (np.full((32, 32, 3), 10+self.steps, dtype=np.uint8),
                3. if self.steps == 2 else -.2, False, self.steps == self.maximum, self._info())

    def close(self):
        self.closed = True


def baseline():
    return {"snake": {"source": "training", "support": 32,
                       "reward_mean": .1, "terminal_rate": .25}}


def test_selected_action_prediction_is_frozen_before_fixture_transition(network, monkeypatch):
    predicted = []
    fixture_steps = [0]
    original = network.model

    def record(hidden, actions):
        if len(actions) == 1:
            predicted.append(fixture_steps[0])
        return original(hidden, actions)

    class OrderedFixture(ToyPixels):
        def step(self, action):
            assert predicted == list(range(self.steps+1))
            result = super().step(action)
            fixture_steps[0] = self.steps
            return result

    monkeypatch.setattr(network, "model", record)
    evaluate(network, 1103, "after_adaptation", "shared", "snake", 810101,
             max_steps=4, training_baselines=baseline(), env_factory=OrderedFixture)
    assert predicted == [0, 1, 2, 3]


def test_real_after_shared_inference_records_every_counterfactual_without_mutation(network, monkeypatch):
    network.train()
    before = {name: value.clone() for name, value in network.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    calls = []
    original = network.step

    def instrumented(pixels, previous_actions, previous_rewards, hidden):
        assert pixels.shape == (1, 3, 32, 32) and pixels.dtype == torch.uint8
        assert previous_actions.shape == previous_rewards.shape == (1,)
        assert not torch.is_grad_enabled() and not network.training
        calls.append((pixels.clone(), previous_actions.clone(), previous_rewards.clone(), hidden.clone()))
        return original(pixels, previous_actions, previous_rewards, hidden)

    monkeypatch.setattr(network, "step", instrumented)
    row, pairs, predictions, trajectory = evaluate(network, 1103, "after_adaptation", "shared", "snake", 810101,
                                                   max_steps=4, training_baselines=baseline(),
                                                   env_factory=ToyPixels, record_trajectory=True)
    assert row["steps"] == len(pairs) == len(predictions) == len(trajectory) == 4
    assert [pair["step"] for pair in pairs] == [1, 2, 3, 4]
    assert row["return"] == pytest.approx(2.4)
    assert row["return_clipped"] == pytest.approx(.4)
    assert row["terminated"] == 0 and row["truncated"] == 1
    assert all(set(("action_default", "action_world_off", "action_gru_reset", "action_pixels_hidden")) <= pair.keys()
               for pair in pairs)
    assert len(json.loads(pairs[0]["q_direct"])) == 10
    assert json.loads(pairs[0]["q_direct"])[2:] == [None]*8
    assert predictions[1]["reward_target_clipped"] == 1.
    assert predictions[-1]["terminal_brier_baseline"] == .25**2
    assert all(prediction["terminated"] == 0 for prediction in predictions)
    # Four forward calls: actual frame, reset-memory counterfactual, zero-pixel
    # counterfactual, actual NEXT frame as detached dynamics target.
    assert len(calls) == 16
    assert torch.count_nonzero(calls[2][0]) == 0
    assert torch.count_nonzero(calls[1][3]) == 0
    assert torch.equal(calls[3][0], torch.full((1, 3, 32, 32), 11, dtype=torch.uint8))
    assert network.training and all(torch.equal(value, before[name]) for name, value in network.state_dict().items())
    assert all(parameter.grad is None for parameter in network.parameters())
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.random.get_rng_state(), rng[2])


@pytest.mark.parametrize("policy", ["world_off", "gru_reset", "pixels_hidden", "untrained", "scratch"])
def test_separate_ablation_episodes_do_not_fabricate_same_state_pairs(network, policy):
    row, pairs, predictions, trajectory = evaluate(network, 1103, "ablation", policy, "snake", 810102,
                    max_steps=3, record_world_predictions=False, env_factory=ToyPixels)
    assert row["policy"] == policy and row["steps"] == 3
    assert pairs == predictions == trajectory == []


def test_random_control_needs_no_network_is_locally_seeded_and_repeatable():
    first = evaluate(None, 0, "baseline", "random", "snake", 810103,
                     max_steps=4, env_factory=ToyPixels, record_trajectory=True)
    second = evaluate(None, 0, "baseline", "random", "snake", 810103,
                      max_steps=4, env_factory=ToyPixels, record_trajectory=True)
    assert first == second
    assert first[1] == first[2] == []
    with pytest.raises(ValueError, match="model_seed=0"):
        evaluate(None, 1103, "baseline", "random", "snake", 810103)


def test_world_holdout_refuses_missing_or_evaluation_derived_baselines_before_environment(network):
    def prohibited(*args, **kwargs):
        pytest.fail("No environment may be started for invalid baseline provenance")

    with pytest.raises(ValueError, match="requires baselines"):
        evaluate(network, 1103, "after_adaptation", "shared", "snake", 810104, env_factory=prohibited)
    wrong = baseline()
    wrong["snake"]["source"] = "evaluation"
    with pytest.raises(ValueError, match="provenance"):
        evaluate(network, 1103, "after_adaptation", "shared", "snake", 810104,
                 training_baselines=wrong, env_factory=prohibited)


def test_world_on_mixes_actual_predictor_values_and_world_off_uses_direct_q(network, monkeypatch):
    q = torch.tensor([[1., 0.]+[99.]*8])
    mask = torch.tensor([[True, True]+[False]*8])
    calls = []

    def controlled_prediction(hidden, supplied_mask, gamma):
        calls.append(gamma)
        return torch.tensor([[0., 10.]+[-float("inf")]*8])

    monkeypatch.setattr(network, "plan_values", controlled_prediction)
    assert _decide(network, q, network.initial(1), mask, world_on=True)[0] == 1
    assert _decide(network, q, network.initial(1), mask, world_on=False)[0] == 0
    assert calls == [.98]


@pytest.mark.parametrize("game", ["snake", "pong", "agar", "flappy", "racing"])
def test_five_actual_simulators_accept_untrained_shared_inference_without_learning(network, game):
    # Five games x <=5 steps = <=25 technical interactions, nonreserved seeds.
    row, pairs, predictions, trajectory = evaluate(network, 1103, "technical_smoke", "shared", game,
                                810109, max_steps=5, record_world_predictions=False)
    assert 1 <= row["steps"] <= 5
    assert row["game"] == game and row["environment_id"].startswith("visual-")
    assert pairs == predictions == trajectory == []


def test_privileged_or_state_dependent_action_mask_is_rejected(network):
    class BadMask(ToyPixels):
        def step(self, action):
            result = list(super().step(action))
            result[4]["action_mask"][1] = False
            return tuple(result)

    with pytest.raises(ValueError, match="mask changed"):
        evaluate(network, 1103, "technical_smoke", "shared", "snake", 810110,
                 max_steps=2, record_world_predictions=False, env_factory=BadMask)


def test_incompatible_controls_fail_before_any_episode(network):
    with pytest.raises(ValueError, match="1..400"):
        evaluate(network, 1103, "technical_smoke", "shared", "snake", 810111, max_steps=401)
    with pytest.raises(ValueError, match="frames require"):
        evaluate(network, 1103, "technical_smoke", "shared", "snake", 810111, include_frames=True)
