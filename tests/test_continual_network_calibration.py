"""Synthetic tensors only: no games, campaign checkpoints or scientific data."""

import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")

from arena.continual.calibration import GateCalibration
from arena.continual.network import ContinualQNetwork


def fixture():
    actions = torch.arange(10).repeat_interleave(8)
    target = torch.full((80, 4), 0.4)
    return dict(
        predicted_h=target.clone().requires_grad_(),
        target_h=target,
        copy_h=torch.zeros_like(target),
        predicted_rewards=torch.zeros(80),
        rewards=torch.zeros(80),
        done_logits=torch.full((80,), -20.0),
        terminated=torch.zeros(80, dtype=torch.bool),
        actions=actions,
        valid=torch.ones(80, dtype=torch.bool),
    )


def mask(*actions):
    value = torch.zeros(1, 10, dtype=torch.bool)
    value[0, list(actions)] = True
    return value


def test_new_residual_can_cross_1_6_without_losing_copy_initialization():
    net = ContinualQNetwork(hidden_size=4, latent_size=4)
    hidden = torch.tensor([[-0.8, 0.8, 0.2, -0.2]])
    predicted, _, _ = net.model(hidden, torch.tensor([0]))
    assert torch.equal(predicted, hidden)
    with torch.no_grad():
        net.transition[-1].bias[0] = math.atanh(0.8)
        net.transition[-1].bias[1] = math.atanh(-0.8)
    changed, _, _ = net.model(hidden, torch.tensor([0]))
    assert changed[0, :2].tolist() == pytest.approx([0.8, -0.8], abs=1e-6)
    assert (changed - hidden)[0, :2].tolist() == pytest.approx([1.6, -1.6], abs=1e-6)
    assert net.architecture_version.endswith("-v2")


def test_control_and_world_optimization_groups_are_exact_and_disjoint():
    net = ContinualQNetwork(hidden_size=4, latent_size=4)
    control = {id(p) for p in net.control_parameters()}
    world = {id(p) for p in net.world_parameters()}
    expected = {
        id(p)
        for module in (net.encoder, net.memory, net.q_head)
        for p in module.parameters()
    }
    assert control == expected
    assert world == {id(p) for p in net.transition.parameters()}
    assert not control & world
    assert control | world == {id(p) for p in net.parameters() if p.requires_grad}
    assert all(not p.requires_grad for p in net.decoder.parameters())


def test_cold_current_good_stale_and_empty_masks():
    gate = GateCalibration()
    assert gate.weights(mask(0, 1), revision=0).item() == 0
    data = fixture()
    gate.refresh(**data, revision=3)
    assert gate.weights(mask(0, 1, 9), revision=3).item() == pytest.approx(0.2)
    assert gate.weights(mask(0, 1), revision=4).item() == 0
    assert gate.weights(mask(), revision=3).item() == 0
    assert data["predicted_h"].grad is None
    assert all(x["support"] == 8 and x["calibrated"] for x in gate.stats)


@pytest.mark.parametrize(
    "field,value",
    [("predicted_h", 0.0), ("predicted_rewards", 1.0), ("done_logits", 0.0)],
)
def test_bad_latent_reward_or_terminal_prediction_turns_the_gate_off(field, value):
    gate = GateCalibration()
    good = fixture()
    gate.refresh(**good, revision=1)
    bad = fixture()
    bad[field] = torch.full_like(bad[field], value)
    gate.refresh(**bad, revision=2)
    assert gate.weights(torch.ones(3, 10, dtype=torch.bool), revision=2).tolist() == [
        0,
        0,
        0,
    ]


def test_every_legal_action_needs_enough_valid_current_support():
    gate = GateCalibration()
    data = fixture()
    data["valid"][-8:] = False  # action9 has no calibration evidence.
    data["valid"][0] = False  # action0 has seven examples, below eight.
    gate.refresh(**data, revision=2)
    masks = torch.cat((mask(1), mask(0, 1), mask(1, 9)))
    assert gate.weights(masks, revision=2).tolist() == pytest.approx([0.2, 0, 0])
    assert gate.stats[0]["support"] == 7
    assert gate.stats[9]["support"] == 0
    # Refresh replaces the previous batch rather than retaining support.
    data["valid"][:] = False
    gate.refresh(**data, revision=3)
    assert gate.weights(mask(1), revision=3).item() == 0


def test_zero_copy_error_cannot_supply_relative_improvement_evidence():
    data = fixture()
    data["copy_h"] = data["target_h"].clone()
    gate = GateCalibration()
    gate.refresh(**data, revision=1)
    assert gate.weights(mask(0), revision=1).item() == 0
    assert gate.stats[0]["mse_ratio"] is None
    assert (
        GateCalibration.from_state_dict(gate.state_dict()).state_dict()
        == gate.state_dict()
    )


def test_json_roundtrip_restores_exact_statistics_revision_and_weights():
    gate = GateCalibration(max_mix=0.15)
    gate.refresh(**fixture(), revision=12)
    state = json.loads(json.dumps(gate.state_dict(), allow_nan=False))
    restored = GateCalibration.from_state_dict(state)
    assert restored.state_dict() == gate.state_dict()
    assert torch.equal(
        restored.weights(mask(1, 8), revision=12), gate.weights(mask(1, 8), revision=12)
    )
    assert restored.weights(mask(1, 8), revision=13).item() == 0
    changed = gate.stats
    changed[1]["calibrated"] = False
    assert gate.stats[1]["calibrated"] is True
    assert (
        GateCalibration.from_state_dict(GateCalibration().state_dict()).revision is None
    )


@pytest.mark.parametrize(
    "change",
    ["negative_support", "forged_pass", "nonfinite", "unknown_key", "bool_revision"],
)
def test_restore_rejects_malformed_or_inconsistent_state(change):
    gate = GateCalibration()
    gate.refresh(**fixture(), revision=1)
    state = copy.deepcopy(gate.state_dict())
    if change == "negative_support":
        state["stats"][0]["support"] = -1
    elif change == "forged_pass":
        state["stats"][0]["calibrated"] = False
    elif change == "nonfinite":
        state["stats"][0]["model_mse"] = float("nan")
    elif change == "unknown_key":
        state["unexpected"] = 1
    else:
        state["revision"] = True
    with pytest.raises(ValueError):
        GateCalibration.from_state_dict(state)


def test_invalid_refresh_fails_closed_and_requires_explicit_contracts():
    gate = GateCalibration()
    gate.refresh(**fixture(), revision=1)
    bad = fixture()
    bad["rewards"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        gate.refresh(**bad, revision=1)
    assert gate.weights(mask(1), revision=1).item() == 0
    with pytest.raises(ValueError, match="older"):
        gate.refresh(**fixture(), revision=0)
    with pytest.raises(ValueError, match="mask"):
        gate.weights(torch.ones(1, 10), revision=1)
    with pytest.raises(ValueError, match="revision"):
        gate.weights(mask(0), revision=True)
