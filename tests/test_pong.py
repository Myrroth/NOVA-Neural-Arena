"""Bounded, synthetic Pong contract tests. No model loading or training."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from arena.envs.pong import PongEnv


def test_pong_gymnasium_contract():
    with PongEnv() as env:
        check_env(env, skip_render_check=True)


def test_observation_actions_and_reset():
    with PongEnv(width=16, height=9) as env:
        obs, info = env.reset(seed=12)
        assert obs.dtype == np.float32
        assert obs.shape == (8,) and env.observation_space.contains(obs)
        assert env.action_space.n == 3
        assert info["steps"] == info["left_score"] == info["right_score"] == 0
        with pytest.raises(ValueError, match="invalid"):
            env.step(99)
        with pytest.raises(ValueError, match="options"):
            env.reset(seed=12, options={"ball": [0, 0]})


@pytest.mark.parametrize("seed", [1, 24, 500001])
def test_seeded_determinism(seed):
    with PongEnv() as a, PongEnv() as b:
        ao, ai = a.reset(seed=seed)
        bo, bi = b.reset(seed=seed)
        np.testing.assert_array_equal(ao, bo)
        assert ai == bi
        for action in [0, 1, 2, 0, 2, 2, 1, 0, 0, 1]:
            x = a.step(action)
            y = b.step(action)
            np.testing.assert_array_equal(x[0], y[0])
            assert x[1:] == y[1:]
            if x[2] or x[3]:
                break


def test_left_paddle_movement_is_bounded():
    with PongEnv(height=9) as env:
        env.reset(seed=2)
        for _ in range(6):
            env.step(1)
        assert env.left_y == 1
        for _ in range(20):
            if env.finished:
                env.reset(seed=2)
            env.step(2)
        assert 1 <= env.left_y <= env.height - 2


def test_left_paddle_reflects_ball():
    with PongEnv() as env:
        env.reset(seed=3)
        env.ball = (2, env.left_y)
        env.velocity = (-1, 0)
        _, reward, terminated, truncated, _ = env.step(0)
        assert env.velocity[0] == 1
        assert env.ball[0] == 2
        assert reward == 0 and not terminated and not truncated


def test_missed_left_paddle_scores_right_and_terminates():
    with PongEnv(target_score=1) as env:
        env.reset(seed=5)
        env.left_y = env.height - 2
        env.ball = (2, 1)
        env.velocity = (-1, 0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == -1.0 and terminated and not truncated
        assert info["right_score"] == 1 and info["winner"] == "right"
        assert env.observation_space.contains(obs)
        with pytest.raises(RuntimeError, match="ended"):
            env.step(0)


def test_missed_right_paddle_scores_left():
    with PongEnv(target_score=1) as env:
        env.reset(seed=9)
        env.right_y = env.height - 2
        env.ball = (env.width - 3, 1)
        env.velocity = (1, 0)
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 1.0 and terminated and not truncated
        assert info["left_score"] == 1 and info["winner"] == "left"


def test_point_below_target_restarts_rally_without_ending():
    with PongEnv(target_score=2) as env:
        env.reset(seed=11)
        env.left_y = env.height - 2
        env.ball = (2, 1)
        env.velocity = (-1, 0)
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == -1.0 and not terminated and not truncated
        assert info["right_score"] == 1
        assert env.ball == (env.width // 2, env.height // 2)


def test_ball_bounces_off_top_wall():
    with PongEnv() as env:
        env.reset(seed=6)
        env.ball = (env.width // 2, 0)
        env.velocity = (-1, -1)
        env.step(0)
        assert env.velocity[1] == 1
        assert env.ball[1] == 1


def test_truncation_is_not_game_victory():
    with PongEnv(max_steps=1) as env:
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated and truncated and info["winner"] is None
        with pytest.raises(RuntimeError, match="ended"):
            env.step(0)


def test_render_and_budget_rejections():
    with PongEnv(render_mode="ansi") as env:
        env.reset(seed=42)
        text = env.render()
        assert "Pong" in text and "|" in text and "O" in text
    for kwargs in [{"width": 9}, {"height": 6}, {"max_steps": 0},
                   {"target_score": 0}, {"render_mode": "rgb_array"}]:
        with pytest.raises(ValueError):
            PongEnv(**kwargs)
