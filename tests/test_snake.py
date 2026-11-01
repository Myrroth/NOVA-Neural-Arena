import numpy as np
import pytest
from arena.envs.snake import SnakeEnv

def test_seed_reproducibility():
    a, b = SnakeEnv(), SnakeEnv()
    assert np.array_equal(a.reset(seed=7)[0], b.reset(seed=7)[0])
    assert a.food == b.food
    for action in [0, 1, 0, 2, 2, 0]:
        x, r1, t1, tr1, i1 = a.step(action)
        y, r2, t2, tr2, i2 = b.step(action)
        np.testing.assert_array_equal(x, y)
        assert (r1, t1, tr1, i1) == (r2, t2, tr2, i2)
        if t1 or tr1:
            break

def test_gymnasium_contract_and_observation():
    env = SnakeEnv(width=8, height=8)
    obs, info = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    assert info == {"score": 0, "length": 3, "steps": 0}
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not terminated and not truncated
    assert info["steps"] == 1

def test_eating_and_growth():
    env = SnakeEnv(width=8, height=8)
    x, y = 4, 4
    env.reset(seed=1, options={"food": (x + 1, y)})
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 1.0
    assert info["score"] == 1 and info["length"] == 4
    assert env.observation_space.contains(obs)
    assert not terminated and not truncated

def test_wall_collision_and_reset_required():
    env = SnakeEnv(width=6, height=6)
    env.reset(seed=0, options={"food": (0, 0)})
    for _ in range(2):
        env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward == -1.0
    assert info["score"] == 0
    with pytest.raises(RuntimeError):
        env.step(0)

def test_truncation():
    env = SnakeEnv(width=6, height=6, max_steps=1)
    env.reset(seed=2, options={"food": (0, 0)})
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["steps"] == 1

def test_invalid_options_actions_and_bounds():
    with pytest.raises(ValueError):
        SnakeEnv(width=5)
    env = SnakeEnv()
    with pytest.raises(ValueError):
        env.reset(seed=1, options={"food": (5, 5)})
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(3)

def test_render():
    env = SnakeEnv(width=6, height=6)
    env.reset(seed=10)
    frame = env.render()
    assert "H" in frame and "F" in frame
    assert len(frame.splitlines()) == 6


def test_official_gymnasium_env_checker():
    checker = pytest.importorskip("gymnasium.utils.env_checker")
    checker.check_env(SnakeEnv(width=8, height=8, max_steps=40))
