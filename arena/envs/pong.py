"""Deterministic, local, single-agent Pong for future reinforcement learning.

Agent controls LEFT paddle. The RIGHT paddle is a fixed environment opponent,
not a policy trained by this project. No historical Snake code is modified.
"""
from __future__ import annotations

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PongEnv(gym.Env):
    """Pong on a discrete board. Actions: 0=stay, 1=up, 2=down.

    Observation (float32, shape 8):
    ball_x/(W-1), ball_y/(H-1), ball_dx, ball_dy,
    left_y/(H-1), right_y/(H-1), left_score/target, right_score/target.
    Velocities are signed -1/0/+1; remaining entries in [0,1].
    Reward: +1 for a point by LEFT, -1 for a point by RIGHT, otherwise 0.
    Reaching the target score terminates. Time limit truncates, never wins.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 12}
    environment_id = "pong-local-v1"
    reward_version = "pong-points-v1"

    def __init__(self, width: int = 20, height: int = 12, *,
                 target_score: int = 3, max_steps: int = 500,
                 render_mode: str | None = None):
        super().__init__()
        if any(type(v) is not int for v in (width, height, target_score, max_steps)):
            raise ValueError("dimensions and budgets must be integers")
        if width < 10 or height < 7 or target_score < 1 or max_steps < 1:
            raise ValueError("width>=10, height>=7, target_score>=1 and max_steps>=1 required")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.width, self.height = width, height
        self.target_score, self.max_steps = target_score, max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.asarray([0, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32),
            high=np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.ball = (width // 2, height // 2)
        self.velocity = (1, 1)
        self.left_y = height // 2
        self.right_y = height // 2
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.finished = True

    def _serve(self) -> None:
        self.ball = (self.width // 2, self.height // 2)
        self.velocity = (
            -1 if int(self.np_random.integers(2)) == 0 else 1,
            -1 if int(self.np_random.integers(2)) == 0 else 1,
        )

    def _obs(self) -> np.ndarray:
        x, y = self.ball
        dx, dy = self.velocity
        return np.asarray((
            x / (self.width - 1), y / (self.height - 1),
            float(dx), float(dy),
            self.left_y / (self.height - 1),
            self.right_y / (self.height - 1),
            self.left_score / self.target_score,
            self.right_score / self.target_score,
        ), dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "steps": self.steps, "left_score": self.left_score,
            "right_score": self.right_score,
            "winner": ("left" if self.left_score >= self.target_score else
                       "right" if self.right_score >= self.target_score else None),
        }

    def reset(self, *, seed: int | None = None,
              options: dict | None = None):
        if options not in (None, {}):
            raise ValueError("reset options/forced game states are not supported")
        super().reset(seed=seed)
        self.left_y = self.right_y = self.height // 2
        self.left_score = self.right_score = self.steps = 0
        self.finished = False
        self._serve()
        return self._obs(), self._info()

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("episode ended; call reset")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid Pong action: {action!r}")
        if action == 1:
            self.left_y = max(1, self.left_y - 1)
        elif action == 2:
            self.left_y = min(self.height - 2, self.left_y + 1)

        # Deterministic opponent is part of the environment, not agent training.
        opponent_target = self.ball[1] if self.velocity[0] > 0 else self.height // 2
        if opponent_target < self.right_y:
            self.right_y = max(1, self.right_y - 1)
        elif opponent_target > self.right_y:
            self.right_y = min(self.height - 2, self.right_y + 1)

        x, y = self.ball
        dx, dy = self.velocity
        next_y = y + dy
        if not 0 <= next_y < self.height:
            dy = -dy
            next_y = y + dy
        next_x = x + dx
        reward = 0.0
        if dx < 0 and next_x <= 1:
            if abs(next_y - self.left_y) <= 1:
                dx, next_x = 1, 2
            else:
                self.right_score += 1
                reward = -1.0
        elif dx > 0 and next_x >= self.width - 2:
            if abs(next_y - self.right_y) <= 1:
                dx, next_x = -1, self.width - 3
            else:
                self.left_score += 1
                reward = 1.0
        point_scored = reward != 0.0
        self.steps += 1
        terminated = max(self.left_score, self.right_score) >= self.target_score
        truncated = self.steps >= self.max_steps and not terminated
        if point_scored and not terminated:
            self._serve()
        elif not point_scored:
            self.ball = (next_x, next_y)
            self.velocity = (dx, dy)
        self.finished = terminated or truncated
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self) -> str:
        board = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for delta in (-1, 0, 1):
            board[self.left_y + delta][1] = "|"
            board[self.right_y + delta][self.width - 2] = "|"
        bx, by = self.ball
        board[by][bx] = "O"
        return (f"Pong left={self.left_score} right={self.right_score} "
                f"step={self.steps}\n" +
                "\n".join("".join(row) for row in board))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
