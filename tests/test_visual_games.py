"""Physics, genuine task outcomes and exact environment restoration, no learner."""
import copy
import json

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from arena.universal.envs import GAMES, make_game


@pytest.mark.parametrize("name", GAMES)
def test_contract_and_seeded_random_stream(name):
    with make_game(name, max_steps=80) as a, make_game(name, max_steps=80) as b:
        check_env(a, skip_render_check=True)
        x, _ = a.reset(seed=312)
        y, _ = b.reset(seed=312)
        assert x.dtype == np.uint8 and x.shape == (32, 32, 3)
        assert np.array_equal(x, y)
        rng = np.random.default_rng(31)
        constant_mask = a.action_mask().copy()
        for _ in range(80):
            action = int(rng.choice(np.flatnonzero(constant_mask)))
            first, second = a.step(action), b.step(action)
            assert np.array_equal(first[0], second[0])
            assert first[1:4] == second[1:4]
            assert np.array_equal(a.action_mask(), constant_mask)
            assert all(np.isfinite(value) for key, value in first[4].items()
                       if isinstance(value, (int, float)))
            if first[2] or first[3]:
                with pytest.raises(RuntimeError):
                    a.step(action)
                break


@pytest.mark.parametrize("name", GAMES)
def test_restore_replays_future_rng_physics_and_metrics(name):
    with make_game(name, max_steps=100) as env:
        env.reset(seed=501)
        controls = list(np.flatnonzero(env.action_mask()))
        env.step(int(controls[0]))
        saved = json.loads(json.dumps(env.snapshot()))
        actions = [int(controls[i % len(controls)]) for i in range(25)]
        expected = []
        for action in actions:
            result = env.step(action)
            expected.append((result, copy.deepcopy(env.snapshot())))
            if result[2] or result[3]:
                break
        env.restore(saved)
        for action, (result, state) in zip(actions, expected):
            actual = env.step(action)
            assert np.array_equal(actual[0], result[0])
            assert actual[1:4] == result[1:4]
            assert env.snapshot() == state


@pytest.mark.parametrize("name", GAMES)
def test_snapshot_cross_game_and_config_rejected(name):
    env = make_game(name, max_steps=50)
    env.reset(seed=11)
    state = env.snapshot()
    wrong = make_game(name, max_steps=51)
    wrong.reset(seed=11)
    with pytest.raises(ValueError, match="mismatch"):
        wrong.restore(state)
    state["environment_id"] = "unrelated-v999"
    with pytest.raises(ValueError, match="mismatch"):
        env.restore(state)


def test_snake_wall_body_and_tail_rules():
    env = make_game("snake")
    env.reset(seed=1)
    s = env.snapshot()
    s["state"].update(body=[[0, 3], [1, 3], [2, 3]], direction=3, food=[7, 7])
    env.restore(s)
    _, reward, done, _, info = env.step(3)
    assert done and reward == -1 and info["wall_collision"] == 1
    env.reset(seed=1)
    s = env.snapshot()
    s["state"].update(body=[[3, 3], [3, 4], [4, 4], [4, 3]], direction=1, food=[7, 7])
    env.restore(s)
    assert not env.step(4)[2]  # entering a tail cell vacated by this move is legal.


def test_pong_return_is_physical_not_free_points():
    env = make_game("pong")
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(ball=[3.1, 16.], velocity=[-.75, .35], left=16.)
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert reward == .15 and not done and info["hits"] == info["incoming"] == 1
    assert env.state["velocity"][0] > 0 and info["points"] == 0


def test_agar_size_determines_eating_not_color_or_label():
    env = make_game("agar")
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[],
                          bots=[[16., 16., 4., 0.]])
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert not done and reward > 0 and info["prey_eaten"] == 1
    env.reset(seed=11)
    state = env.snapshot()
    state["state"].update(player=[16., 16., 10.], pellets=[],
                          bots=[[16., 16., 25., 0.]])
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert done and reward == -1 and info["eaten"] == 1


def test_flap_changes_vertical_motion_and_pipe_collision_is_real():
    a, b = make_game("flappy"), make_game("flappy")
    a.reset(seed=31); b.reset(seed=31)
    a.step(9); b.step(0)
    assert a.state["y"] < b.state["y"] and a.state["vy"] < 0
    s = b.snapshot()
    s["state"].update(y=3., vy=0., pipes=[[8., 20., False], [26., 16., False]])
    b.restore(s)
    assert b.step(0)[2]


def test_racing_throttle_brake_grip_and_offroad():
    env = make_game("racing")
    env.reset(seed=11)
    initial = env.state["speed"]
    env.step(1)
    assert env.state["speed"] > initial
    env.step(2)
    assert env.state["speed"] < initial
    state = env.snapshot()
    state["state"]["x"] = 0.
    env.restore(state)
    _, reward, done, _, info = env.step(0)
    assert done and reward == -1 and info["collision"] == 1


def test_limits_are_truncations_not_wins():
    env = make_game("racing", max_steps=1)
    env.reset(seed=11)
    _, _, done, truncated, info = env.step(0)
    assert not done and truncated and info["terminal_reason"] == "step_limit"
