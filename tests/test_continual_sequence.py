"""Synthetic tensor equivalence for the frame-batched continual encoder.

No games, checkpoints, optimizers or performance campaign are used. The
reference is an explicit causal loop through the unchanged public step API.
"""

import copy

import pytest
import torch
from torch import nn

from arena.continual.network import ContinualQNetwork
from arena.universal.network import VisualQNetwork


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def network():
    with torch.random.fork_rng():
        torch.manual_seed(831291)
        return ContinualQNetwork(hidden_size=8, latent_size=8)


def inputs(batch=3, time=24):
    generator = torch.Generator().manual_seed(831292)
    pixels = torch.randint(
        0, 256, (batch, time, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    actions = torch.randint(0, 10, (batch, time), generator=generator)
    rewards = torch.rand(batch, time, generator=generator) * 4 - 2
    hidden = torch.rand(batch, 8, generator=generator) - 0.5
    resets = torch.zeros(batch, time, dtype=torch.bool)
    for row in range(batch):
        resets[row, row % time] = True
        resets[row, 8::16] = True
    lengths = torch.tensor([max(1, time - 3 * row) for row in range(batch)])
    valid = torch.arange(time)[None, :] < lengths[:, None]
    # Padded replay frames remain in the sequence. Validity affects the loss;
    # it must not silently change how sequence() handles explicit reset inputs.
    pixels[~valid] = 0
    actions[~valid] = 0
    rewards[~valid] = 0
    resets[~valid] = True
    return pixels, actions, rewards, hidden, resets, valid


def reference_loop(net, pixels, actions, rewards, hidden=None, resets=None):
    batch, time = pixels.shape[:2]
    state = net.initial(batch, pixels.device) if hidden is None else hidden
    if resets is None:
        resets = torch.zeros(batch, time, dtype=torch.bool, device=pixels.device)
    logits, states, latents = [], [], []
    for step in range(time):
        state = torch.where(resets[:, step, None], torch.zeros_like(state), state)
        q, state, latent = net.step(
            pixels[:, step], actions[:, step], rewards[:, step], state
        )
        logits.append(q)
        states.append(state)
        latents.append(latent)
    return torch.stack(logits, 1), torch.stack(states, 1), torch.stack(latents, 1)


def assert_outputs_close(actual, expected):
    for candidate, reference in zip(actual, expected):
        # Larger convolution batches can use a different float32 reduction
        # order while retaining the same per-frame mathematical operation.
        torch.testing.assert_close(candidate, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "batch,time,explicit", [(1, 1, False), (3, 24, False), (3, 24, True)]
)
@pytest.mark.parametrize("training", [False, True])
def test_outputs_hidden_and_latents_match_explicit_steps(
    network, batch, time, explicit, training
):
    network.train(training)
    pixels, actions, rewards, hidden, resets, _ = inputs(batch, time)
    kwargs = {"hidden": hidden, "resets": resets} if explicit else {}
    assert_outputs_close(
        network.sequence(pixels, actions, rewards, **kwargs),
        reference_loop(network, pixels, actions, rewards, **kwargs),
    )


def test_strided_pixels_previous_inputs_and_resets_preserve_the_same_math(network):
    pixels, actions, rewards, hidden, resets, _ = inputs()
    pixels = pixels.transpose(-1, -2)
    actions = actions.T.contiguous().T
    rewards = rewards.T.contiguous().T
    resets = resets.T.contiguous().T
    assert not any(
        tensor.is_contiguous() for tensor in (pixels, actions, rewards, resets)
    )
    assert_outputs_close(
        network.sequence(pixels, actions, rewards, hidden, resets),
        reference_loop(network, pixels, actions, rewards, hidden, resets),
    )


def burn_then_unroll(net, function, pixels, actions, rewards, hidden, resets, burn):
    if burn:
        with torch.no_grad():
            _, history, _ = function(
                net,
                pixels[:, :burn],
                actions[:, :burn],
                rewards[:, :burn],
                hidden,
                resets[:, :burn],
            )
        hidden = history[:, -1].detach()
    return function(
        net,
        pixels[:, burn:],
        actions[:, burn:],
        rewards[:, burn:],
        hidden,
        resets[:, burn:],
    )


def vectorized(net, *args):
    return net.sequence(*args)


def synthetic_loss(net, outputs, actions, valid):
    q, states, latents = outputs
    predicted_h, predicted_r, terminal_logits = net.model(
        states.reshape(-1, net.hidden_size), actions.reshape(-1)
    )
    predicted_h = predicted_h.reshape_as(states)
    predicted_r = predicted_r.reshape(valid.shape)
    terminal_logits = terminal_logits.reshape(valid.shape)
    loss_per_frame = (
        q.square().mean(-1)
        + 0.3 * states.square().mean(-1)
        + 0.2 * latents.square().mean(-1)
        + 0.4 * (predicted_h - 0.2).square().mean(-1)
        + 0.1 * (predicted_r + 0.3).square()
        + 0.2 * (terminal_logits - 0.4).square()
    )
    return loss_per_frame[valid].mean(), (predicted_h, predicted_r, terminal_logits)


@pytest.mark.parametrize("burn", [0, 15])
def test_burn_resets_padding_world_logits_and_autograd_match_reference(network, burn):
    reference = copy.deepcopy(network)
    pixels, actions, rewards, hidden, resets, valid = inputs()
    actual_hidden = hidden.clone().requires_grad_()
    reference_hidden = hidden.clone().requires_grad_()
    actual_rewards = rewards.clone().requires_grad_()
    reference_rewards = rewards.clone().requires_grad_()
    actual = burn_then_unroll(
        network,
        vectorized,
        pixels,
        actions,
        actual_rewards,
        actual_hidden,
        resets,
        burn,
    )
    expected = burn_then_unroll(
        reference,
        reference_loop,
        pixels,
        actions,
        reference_rewards,
        reference_hidden,
        resets,
        burn,
    )
    assert_outputs_close(actual, expected)
    loss, actual_predictions = synthetic_loss(
        network, actual, actions[:, burn:], valid[:, burn:]
    )
    expected_loss, expected_predictions = synthetic_loss(
        reference, expected, actions[:, burn:], valid[:, burn:]
    )
    assert_outputs_close(actual_predictions, expected_predictions)
    torch.testing.assert_close(loss, expected_loss, atol=1e-6, rtol=1e-5)
    loss.backward()
    expected_loss.backward()
    for (name, parameter), (reference_name, reference_parameter) in zip(
        network.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                atol=1e-6,
                rtol=1e-5,
                msg=lambda message: f"{name}: {message}",
            )
    for actual_input, reference_input in (
        (actual_hidden, reference_hidden),
        (actual_rewards, reference_rewards),
    ):
        if reference_input.grad is None:
            assert actual_input.grad is None
        else:
            torch.testing.assert_close(
                actual_input.grad, reference_input.grad, atol=1e-6, rtol=1e-5
            )
    if burn:
        assert actual_hidden.grad is None
        assert torch.count_nonzero(actual_rewards.grad[:, :burn]) == 0
    assert all(parameter.grad is None for parameter in network.decoder.parameters())


def test_frozen_teacher_sequence_matches_steps_without_state_mode_or_rng_changes(
    network,
):
    teacher = copy.deepcopy(network).requires_grad_(False).eval()
    pixels, actions, rewards, hidden, resets, _ = inputs()
    before = {name: tensor.clone() for name, tensor in teacher.state_dict().items()}
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        actual = teacher.sequence(pixels, actions, rewards, hidden, resets)
        expected = reference_loop(teacher, pixels, actions, rewards, hidden, resets)
    assert_outputs_close(actual, expected)
    assert all(not tensor.requires_grad for tensor in actual)
    assert all(not module.training for module in teacher.modules())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        torch.equal(before[name], tensor)
        for name, tensor in teacher.state_dict().items()
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_future_observations_inputs_and_resets_cannot_change_earlier_outputs(network):
    pixels, actions, rewards, hidden, resets, _ = inputs()
    cutoff = 9
    with torch.no_grad():
        before = network.sequence(pixels, actions, rewards, hidden, resets)
        changed_pixels, changed_actions, changed_rewards, changed_resets = (
            tensor.clone() for tensor in (pixels, actions, rewards, resets)
        )
        changed_pixels[:, cutoff:] = 255 - changed_pixels[:, cutoff:]
        changed_actions[:, cutoff:] = (changed_actions[:, cutoff:] + 3) % 10
        changed_rewards[:, cutoff:] = -changed_rewards[:, cutoff:]
        changed_resets[:, cutoff:] = ~changed_resets[:, cutoff:]
        after = network.sequence(
            changed_pixels, changed_actions, changed_rewards, hidden, changed_resets
        )
    for actual, expected in zip(after, before):
        torch.testing.assert_close(
            actual[:, :cutoff], expected[:, :cutoff], atol=0, rtol=0
        )
    assert not torch.equal(after[1][:, cutoff:], before[1][:, cutoff:])


def test_other_batch_members_cannot_influence_a_sequence_even_in_training_mode(network):
    pixels, actions, rewards, hidden, resets, _ = inputs()
    network.train()
    before = network.sequence(pixels, actions, rewards, hidden, resets)
    pixels[1:] = 255 - pixels[1:]
    actions[1:] = (actions[1:] + 1) % 10
    rewards[1:] *= -1
    hidden[1:] *= -1
    resets[1:] = ~resets[1:]
    after = network.sequence(pixels, actions, rewards, hidden, resets)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)


def test_reset_blocks_history_gradient_but_keeps_explicit_current_previous_reward(
    network,
):
    pixels, actions, _, hidden, _, _ = inputs(batch=1, time=6)
    hidden.requires_grad_()
    rewards = torch.full((1, 6), 0.2, requires_grad=True)
    resets = torch.zeros(1, 6, dtype=torch.bool)
    resets[0, 3] = True
    q, states, _ = network.sequence(pixels, actions, rewards, hidden, resets)
    (q[:, 3:].sum() + states[:, 3:].sum()).backward()
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(rewards.grad[:, :3]) == 0
    assert torch.count_nonzero(rewards.grad[:, 3:]) > 0


def test_one_encoder_batch_replaces_the_per_timestep_encoder_loop(network):
    pixels, actions, rewards, hidden, resets, _ = inputs(batch=2, time=17)
    encoder_shapes = []
    memory_shapes = []
    encoder_hook = network.encoder.register_forward_pre_hook(
        lambda module, args: encoder_shapes.append(tuple(args[0].shape))
    )
    memory_hook = network.memory.register_forward_pre_hook(
        lambda module, args: memory_shapes.append(tuple(args[0].shape))
    )
    try:
        with torch.no_grad():
            actual = network.sequence(pixels, actions, rewards, hidden, resets)
            assert encoder_shapes == [(34, 3, 32, 32)]
            assert memory_shapes == [(2, network.latent_size + 11)] * 17
            encoder_shapes.clear()
            memory_shapes.clear()
            reference = reference_loop(
                network, pixels, actions, rewards, hidden, resets
            )
            assert encoder_shapes == [(2, 3, 32, 32)] * 17
            assert memory_shapes == [(2, network.latent_size + 11)] * 17
        assert_outputs_close(actual, reference)
    finally:
        encoder_hook.remove()
        memory_hook.remove()


def test_encoder_contract_is_frame_local_and_public_step_is_unchanged(network):
    assert ContinualQNetwork.step is VisualQNetwork.step
    allowed = (
        nn.Sequential,
        nn.Conv2d,
        nn.ReLU,
        nn.Flatten,
        nn.Linear,
        nn.LayerNorm,
        nn.Tanh,
    )
    assert all(isinstance(module, allowed) for module in network.encoder.modules())
    for module in network.encoder.modules():
        if isinstance(module, nn.LayerNorm):
            assert module.normalized_shape == (network.latent_size,)


@pytest.mark.parametrize(
    "invalid",
    [
        "pixels_dtype",
        "pixels_shape",
        "empty_time",
        "actions_dtype",
        "actions_range",
        "actions_shape",
        "reward_dtype",
        "reward_nan",
        "resets_dtype",
        "resets_shape",
        "hidden_batch",
        "hidden_shape",
    ],
)
def test_existing_input_contracts_are_preserved(network, invalid):
    pixels, actions, rewards, hidden, resets, _ = inputs(batch=2, time=4)
    if invalid == "pixels_dtype":
        pixels = pixels.float()
    elif invalid == "pixels_shape":
        pixels = pixels[..., :16]
    elif invalid == "empty_time":
        pixels = pixels[:, :0]
    elif invalid == "actions_dtype":
        actions = actions.float()
    elif invalid == "actions_range":
        actions[0, -1] = 10
    elif invalid == "actions_shape":
        actions = actions[:, :-1]
    elif invalid == "reward_dtype":
        rewards = rewards.long()
    elif invalid == "reward_nan":
        rewards[0, -1] = torch.nan
    elif invalid == "resets_dtype":
        resets = resets.long()
    elif invalid == "resets_shape":
        resets = resets[:, :-1]
    elif invalid == "hidden_batch":
        hidden = hidden[:1]
    elif invalid == "hidden_shape":
        hidden = hidden[:, :4]
    with pytest.raises(ValueError):
        network.sequence(pixels, actions, rewards, hidden, resets)
