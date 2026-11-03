"""Small tensor-only checks; no prepared-game episodes or training campaign."""
import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from arena.universal.network import VisualQNetwork


def inputs(batch=2, time=4):
    generator = torch.Generator().manual_seed(907)
    pixels = torch.randint(0, 256, (batch, time, 3, 32, 32),
                           dtype=torch.uint8, generator=generator)
    actions = torch.randint(0, 10, (batch, time), generator=generator)
    rewards = torch.randn(batch, time, generator=generator)
    return pixels, actions, rewards


def test_one_shared_visual_recurrent_policy_has_no_game_input_or_heads():
    network = VisualQNetwork()
    pixels, actions, rewards = inputs()
    q, state, latent = network.step(pixels[:, 0], actions[:, 0], rewards[:, 0], network.initial(2))
    assert q.shape == (2, 10)
    assert state.shape == latent.shape == (2, 64)
    assert torch.equal(q, network.q_head(state))
    assert "game" not in inspect.signature(network.step).parameters
    assert not any(isinstance(module, nn.ModuleDict) for module in network.modules())
    assert sum(parameter.numel() for parameter in network.parameters()) < 500_000
    assert all(torch.isfinite(value).all() for value in (q, state, latent))


def test_sequence_matches_step_and_reset_removes_prior_hidden_information():
    network = VisualQNetwork()
    pixels, actions, rewards = inputs(time=5)
    resets = torch.zeros(2, 5, dtype=torch.bool)
    resets[:, 2] = True
    initial = torch.ones(2, 64)
    q, states, latents = network.sequence(pixels, actions, rewards, initial, resets)
    state, manual_q, manual_states, manual_latents = initial, [], [], []
    for t in range(5):
        state = torch.where(resets[:, t, None], torch.zeros_like(state), state)
        values, state, latent = network.step(pixels[:, t], actions[:, t], rewards[:, t], state)
        manual_q.append(values)
        manual_states.append(state)
        manual_latents.append(latent)
    torch.testing.assert_close(q, torch.stack(manual_q, dim=1))
    torch.testing.assert_close(states, torch.stack(manual_states, dim=1))
    torch.testing.assert_close(latents, torch.stack(manual_latents, dim=1))
    other_q, other_states, _ = network.sequence(pixels, actions, rewards, -initial, resets)
    torch.testing.assert_close(q[:, 2:], other_q[:, 2:])
    torch.testing.assert_close(states[:, 2:], other_states[:, 2:])
    # A reset blocks gradient flow to pre-episode memory as well as its values.
    original = network.initial(2).requires_grad_()
    reset_values, _, _ = network.sequence(pixels, actions, rewards, original, resets)
    reset_values[:, 2:].sum().backward()
    assert original.grad is not None
    assert torch.count_nonzero(original.grad) == 0


def test_world_model_is_action_conditioned_and_decoder_supports_gradients():
    network = VisualQNetwork()
    pixels, actions, rewards = inputs()
    q, state, latent = network.step(pixels[:, 0], actions[:, 0], rewards[:, 0], network.initial(2))
    repeated = state[:1].expand(10, -1)
    predicted, predicted_reward, done_logits = network.model(repeated, torch.arange(10))
    assert predicted.shape == (10, 64)
    assert predicted_reward.shape == done_logits.shape == (10,)
    # The zero-initialized residual starts at a persistence baseline, while the
    # learned model still receives and can distinguish the chosen action.
    torch.testing.assert_close(predicted, repeated)
    assert not torch.allclose(predicted_reward[0], predicted_reward[1])
    decoded = network.decode(latent)
    assert decoded.shape == (2, 3, 8, 8)
    assert bool(((decoded >= 0) & (decoded <= 1)).all())
    loss = q.square().mean() + predicted.square().mean()
    loss = loss + predicted_reward.square().mean() + done_logits.square().mean() + decoded.mean()
    loss.backward()
    for module in (network.encoder, network.memory, network.q_head,
                   network.transition, network.decoder):
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   and bool((p.grad != 0).any()) for p in module.parameters())


def test_planning_uses_same_q_head_masks_and_predicted_terminal_gate(monkeypatch):
    network = VisualQNetwork()
    network.q_head = nn.Linear(64, 10, bias=False)
    with torch.no_grad():
        network.q_head.weight.zero_()
        network.q_head.weight[:, 0] = torch.arange(10)

    def known_dynamics(hidden, actions):
        next_hidden = torch.zeros_like(hidden)
        next_hidden[:, 0] = 1
        reward = actions.float() + .5
        terminal = torch.where(actions == 4, 100.0, -100.0)
        return next_hidden, reward, terminal

    monkeypatch.setattr(network, "model", known_dynamics)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, [0, 1, 4]] = True
    values = network.plan_values(network.initial(2), mask, gamma=.98)
    # Max next Q is 4, not the unavailable input 9.
    torch.testing.assert_close(values[:, 0], torch.full((2,), .5 + .98 * 4))
    torch.testing.assert_close(values[:, 1], torch.full((2,), 1.5 + .98 * 4))
    # A predicted terminal state does not bootstrap.
    torch.testing.assert_close(values[:, 4], torch.full((2,), 4.5))
    assert torch.isneginf(values[~mask]).all()


def test_bad_pixel_action_or_mask_contract_fails_early():
    network = VisualQNetwork()
    pixels, actions, rewards = inputs()
    with pytest.raises(ValueError, match="uint8"):
        network.step(pixels[:, 0].float(), actions[:, 0], rewards[:, 0], network.initial(2))
    with pytest.raises(ValueError, match="indices"):
        network.model(network.initial(2), torch.tensor([-1, 10]))
    with pytest.raises(ValueError, match="legal input"):
        network.plan_values(network.initial(2), torch.zeros(2, 10, dtype=torch.bool))
    with pytest.raises(ValueError, match="resets"):
        network.sequence(pixels, actions, rewards, resets=torch.zeros(2, 4))


def test_previous_reward_transform_is_identical_for_raw_and_clipped_replay_values():
    network = VisualQNetwork()
    pixels, actions, _ = inputs()
    state = network.initial(2)
    raw = network.step(pixels[:, 0], actions[:, 0], torch.tensor([20.0, -7.0]), state)
    clipped = network.step(pixels[:, 0], actions[:, 0], torch.tensor([1.0, -1.0]), state)
    for actual, expected in zip(raw, clipped):
        torch.testing.assert_close(actual, expected)
