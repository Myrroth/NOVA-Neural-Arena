"""Frozen inference on synthetic images only; no simulator or official model."""

import copy
import random

import numpy as np
import pytest
import torch

from arena.continual.calibration import GateCalibration
from arena.continual.network import ContinualQNetwork
from arena.continual.policy import FrozenPolicyController
from arena.continual.trainer import ContinualConfig, ContinualTrainer


class MaskOnly:
    def __init__(self, mask):
        self._mask = mask

    def action_mask(self):
        return self._mask.copy()


def network():
    return ContinualQNetwork(hidden_size=8, latent_size=8).requires_grad_(False).eval()


def good_calibration(revision=0, *, unsupported=None):
    count = 80
    h = torch.full((count, 8), 0.4)
    valid = torch.ones(count, dtype=torch.bool)
    if unsupported is not None:
        valid[8 * unsupported : 8 * (unsupported + 1)] = False
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
        valid,
        revision=revision,
    )
    return gate


def image(rng=None):
    rng = np.random.default_rng(831091) if rng is None else rng
    return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)


def legal(*indices):
    mask = np.zeros(10, dtype=bool)
    mask[list(indices)] = True
    return mask


def test_controller_matches_trainer_actor_context_and_gated_actions():
    cfg = ContinualConfig(
        hidden_size=8, latent_size=8, context_length=4, gamma=0.91, threads=1
    )
    trainer = ContinualTrainer(831092, cfg)
    trainer.online.requires_grad_(False).eval()
    trainer.calibration = good_calibration()
    mask = legal(0, 1, 3, 4, 8)
    controller = FrozenPolicyController(
        trainer.online,
        calibration=trainer.calibration,
        policy_revision=0,
        context_length=4,
        gamma=cfg.gamma,
    )
    rng = np.random.default_rng(831093)
    actor = {"env": MaskOnly(mask), "history": []}
    for step in range(21):
        observation = image(rng)
        pa, pr = step % 10, float(rng.uniform(-1, 1))
        actor.update(observation=observation, previous_action=pa, previous_reward=pr)
        expected, history, hidden = trainer._observe_actor(actor)
        result = controller.act(observation, pa, pr, mask)
        torch.testing.assert_close(result["q_mixed"], expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(result["hidden"], hidden, rtol=0, atol=1e-6)
        assert result["action"] == int(expected.argmax())
        assert result["context_reset"] == (step % 4 == 0)
        assert result["planning_weight"] == pytest.approx(
            0 if (step + 1) % 4 == 0 else 0.2
        )
        actor.update(history=history, _cached_hidden=hidden, _cached_revision=0)


@pytest.mark.parametrize("condition", ["cold", "stale", "unsupported", "world_off"])
def test_zero_mix_is_exact_direct_and_never_calls_the_model(condition, monkeypatch):
    net = network()
    gate = (
        GateCalibration()
        if condition == "cold"
        else good_calibration(
            revision=1 if condition == "stale" else 0,
            unsupported=9 if condition == "unsupported" else None,
        )
    )
    controller = FrozenPolicyController(net, calibration=gate, policy_revision=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unvalidated or disabled lookahead was called")

    monkeypatch.setattr(net, "plan_values", forbidden)
    result = controller.act(
        image(), 0, 0.0, legal(0, 9), world_on=condition != "world_off"
    )
    assert result["planning_weight"] == 0
    assert result["model_values"] is None
    assert torch.equal(result["q_direct"], result["q_mixed"])
    assert not torch.isnan(result["q_mixed"]).any()


def test_next_periodic_reset_blocks_lookahead_only_at_original_phase(monkeypatch):
    net = network()
    controller = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0, context_length=2
    )
    first = controller.act(image(), 0, 0.0, legal(0, 1))
    assert first["planning_weight"] == pytest.approx(0.2)
    assert 0 < first["planning_weight"] <= 0.2

    def forbidden(*args, **kwargs):
        raise AssertionError("lookahead must not bridge the scheduled reset")

    monkeypatch.setattr(net, "plan_values", forbidden)
    second = controller.act(image(), first["action"], 0.1, legal(0, 1))
    assert not second["context_reset"] and second["planning_weight"] == 0
    assert torch.equal(second["q_direct"], second["q_mixed"])


def test_pixels_and_gru_ablations_change_only_their_declared_inputs():
    net = network()
    controller = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0
    )
    frame = image()
    controller.act(frame, 0, 0.0, legal(0, 1))
    hidden_pixels = controller.act(
        frame, 1, 0.2, legal(0, 1), pixels_hidden=True, world_on=False
    )
    with torch.inference_mode():
        expected, _, _ = net.step(
            torch.zeros(1, 3, 32, 32, dtype=torch.uint8),
            torch.tensor([1]),
            torch.tensor([0.2]),
            hidden_pixels["hidden_before"],
        )
    torch.testing.assert_close(
        hidden_pixels["q_direct"][:2], expected[0, :2], rtol=0, atol=0
    )
    reset = controller.act(frame, 1, 0.2, legal(0, 1), gru_reset=True)
    assert reset["context_reset"] and torch.count_nonzero(reset["hidden_before"]) == 0
    # This ablation is not silently changed into world_off.
    assert reset["planning_weight"] == pytest.approx(0.2)
    assert reset["model_values"] is not None


def test_world_off_preserves_direct_q_and_memory_given_identical_own_inputs():
    net = network()
    a = FrozenPolicyController(net, calibration=good_calibration(), policy_revision=0)
    b = FrozenPolicyController(net, calibration=good_calibration(), policy_revision=0)
    rng = np.random.default_rng(831094)
    for step in range(5):
        frame = image(rng)
        full = a.act(frame, step, 0.2, legal(0, 1, 4))
        direct = b.act(frame, step, 0.2, legal(0, 1, 4), world_on=False)
        assert torch.equal(full["q_direct"], direct["q_direct"])
        assert torch.equal(full["hidden"], direct["hidden"])
        assert torch.equal(direct["q_direct"], direct["q_mixed"])


def test_reset_copies_calibration_and_returned_tensors_do_not_alias_history():
    net = network()
    gate = good_calibration()
    controller = FrozenPolicyController(net, calibration=gate, policy_revision=0)
    other = FrozenPolicyController(net, calibration=gate, policy_revision=0)
    gate.refresh(
        **{  # An external refresh cannot mutate the frozen controller.
            "predicted_h": torch.zeros(1, 8),
            "target_h": torch.ones(1, 8),
            "copy_h": torch.zeros(1, 8),
            "predicted_rewards": torch.zeros(1),
            "rewards": torch.zeros(1),
            "done_logits": torch.zeros(1),
            "terminated": torch.zeros(1, dtype=torch.bool),
            "actions": torch.zeros(1, dtype=torch.long),
            "valid": torch.ones(1, dtype=torch.bool),
        },
        revision=0,
    )
    frame = image()
    first = controller.act(frame, 0, 0.0, legal(0, 1))
    assert first["planning_weight"] == pytest.approx(0.2)
    other.act(frame, 0, 0.0, legal(0, 1))
    first["hidden"].zero_()
    first["hidden_before"].fill_(1)
    actual = controller.act(frame, 1, 0.1, legal(0, 1))
    expected = other.act(frame, 1, 0.1, legal(0, 1))
    assert torch.equal(actual["hidden"], expected["hidden"])
    controller.reset()
    reset = controller.act(frame, 0, 0.0, legal(0, 1))
    fresh = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0
    ).act(frame, 0, 0.0, legal(0, 1))
    assert reset["context_reset"] and torch.equal(reset["q_mixed"], fresh["q_mixed"])


def test_inference_preserves_global_rng_modes_parameters_and_existing_gradients():
    net = network()
    first_parameter = next(net.parameters())
    first_parameter.grad = torch.ones_like(first_parameter)
    saved_grad = first_parameter.grad.clone()
    states = {name: value.clone() for name, value in net.state_dict().items()}
    python = random.getstate()
    numpy = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    controller = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0
    )
    for i in range(6):
        controller.act(image(), i, 0.3, legal(0, 1, 3), gru_reset=i % 2 == 0)
    assert random.getstate() == python
    now = np.random.get_state()
    assert (
        now[0] == numpy[0] and np.array_equal(now[1], numpy[1]) and now[2:] == numpy[2:]
    )
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert all(
        torch.equal(value, net.state_dict()[name]) for name, value in states.items()
    )
    assert torch.equal(first_parameter.grad, saved_grad)
    assert all(not p.requires_grad for p in net.parameters()) and all(
        not m.training for m in net.modules()
    )


def test_frozen_contract_rejects_trainable_train_mode_and_later_weight_changes():
    trainable = ContinualQNetwork(hidden_size=8, latent_size=8).eval()
    with pytest.raises(ValueError, match="frozen"):
        FrozenPolicyController(
            trainable, calibration=GateCalibration(), policy_revision=0
        )
    net = network().train()
    with pytest.raises(ValueError, match="eval"):
        FrozenPolicyController(net, calibration=GateCalibration(), policy_revision=0)
    net.eval()
    controller = FrozenPolicyController(
        net, calibration=GateCalibration(), policy_revision=0
    )
    with torch.no_grad():
        next(net.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="changed"):
        controller.act(image(), 0, 0.0, legal(0))


def test_illegal_high_q_is_masked_and_ties_use_first_legal_action():
    net = ContinualQNetwork(hidden_size=8, latent_size=8)
    with torch.no_grad():
        net.q_head[-1].weight.zero_()
        net.q_head[-1].bias.zero_()
        net.q_head[-1].bias[9] = 1000
    net.requires_grad_(False).eval()
    result = FrozenPolicyController(
        net, calibration=GateCalibration(), policy_revision=0
    ).act(image(), 0, 0.0, legal(1, 4))
    assert result["action"] == 1
    assert torch.isneginf(result["q_direct"][[0, 2, 3, 5, 6, 7, 8, 9]]).all()
    assert torch.equal(result["q_direct"], result["q_mixed"])


@pytest.mark.parametrize("bad", ["frame", "action", "reward", "mask", "flag"])
def test_invalid_inputs_do_not_advance_periodic_state(bad):
    net = network()
    controller = FrozenPolicyController(
        net, calibration=GateCalibration(), policy_revision=0, context_length=2
    )
    args = [image(), 0, 0.0, legal(0)]
    kwargs = {}
    if bad == "frame":
        args[0] = args[0].astype(np.float32)
    elif bad == "action":
        args[1] = True
    elif bad == "reward":
        args[2] = float("nan")
    elif bad == "mask":
        args[3] = np.ones(10, dtype=np.int64)
    else:
        kwargs["gru_reset"] = 1
    with pytest.raises(ValueError):
        controller.act(*args, **kwargs)
    assert controller.act(image(), 0, 0.0, legal(0))["context_reset"]


def test_context_and_revision_contracts_are_explicit():
    net = network()
    for kwargs in (
        {"policy_revision": True},
        {"policy_revision": -1},
        {"policy_revision": 0, "context_length": 1},
        {"policy_revision": 0, "gamma": float("nan")},
    ):
        with pytest.raises(ValueError):
            FrozenPolicyController(net, calibration=GateCalibration(), **kwargs)


def test_fork_preserves_phase_and_decisions_without_aliasing_state_or_using_rng():
    net = network()
    controller = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0, context_length=4
    )
    frame = image()
    controller.act(frame, 0, 0.0, legal(0, 1))
    controller.act(frame, 1, 0.2, legal(0, 1))
    original_hidden = controller._hidden.clone()
    original_calibration = controller._calibration.state_dict()
    python = random.getstate()
    numpy = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    forked = controller.fork()
    assert forked.network is controller.network
    assert forked._calibration is not controller._calibration
    assert forked._calibration.state_dict() == original_calibration
    assert forked._hidden.data_ptr() != controller._hidden.data_ptr()
    assert forked._observations == controller._observations == 2
    assert random.getstate() == python
    now = np.random.get_state()
    assert (
        now[0] == numpy[0] and np.array_equal(now[1], numpy[1]) and now[2:] == numpy[2:]
    )
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    actual = forked.act(frame, 1, 0.1, legal(0, 1))
    assert controller._observations == 2 and torch.equal(
        controller._hidden, original_hidden
    )
    expected = controller.act(frame, 1, 0.1, legal(0, 1))
    assert actual["action"] == expected["action"]
    assert torch.equal(actual["hidden"], expected["hidden"])
    assert torch.equal(actual["q_mixed"], expected["q_mixed"])
    forked.reset()
    assert controller._observations == 3
    assert torch.equal(controller._hidden, expected["hidden"])


def test_fork_rejects_a_network_that_changed_after_freezing():
    net = network()
    controller = FrozenPolicyController(
        net, calibration=GateCalibration(), policy_revision=0
    )
    with torch.no_grad():
        next(net.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="changed"):
        controller.fork()


def test_fork_reuses_validated_network_without_constructor_or_finite_scan(monkeypatch):
    net = network()
    controller = FrozenPolicyController(
        net, calibration=good_calibration(), policy_revision=0
    )
    controller.act(image(), 0, 0.0, legal(0, 1))
    original_stamp = FrozenPolicyController._network_stamp
    checks = []

    def tracked_stamp(self, *, check_finite=False):
        checks.append(check_finite)
        if check_finite:
            raise AssertionError("fork repeated the construction-time weight scan")
        return original_stamp(self, check_finite=check_finite)

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("fork called the policy constructor again")

    monkeypatch.setattr(FrozenPolicyController, "_network_stamp", tracked_stamp)
    monkeypatch.setattr(FrozenPolicyController, "__init__", forbidden_constructor)
    forked = controller.fork()
    assert checks == [False, False]
    assert forked.network is net and forked._initial_stamp is controller._initial_stamp
    assert forked._hidden.data_ptr() != controller._hidden.data_ptr()
    assert forked._calibration is not controller._calibration
    with torch.no_grad():
        next(net.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="changed"):
        controller.fork()
    with pytest.raises(ValueError, match="changed"):
        forked.act(image(), 1, 0.1, legal(0, 1))
