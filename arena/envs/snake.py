"""Minimal deterministic Snake with relative actions and Gymnasium API."""
from __future__ import annotations
from collections import deque
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))

class SnakeEnv(gym.Env):
    """Actions: straight (0), turn right (1), turn left (2).
    11 features: danger straight/right/left, direction one-hot and relative food.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 8}

    def __init__(self, width: int = 10, height: int = 10, max_steps: int = 200, render_mode: str | None = None):
        super().__init__()
        if width < 6 or height < 6 or max_steps < 1:
            raise ValueError("width and height must be >= 6; max_steps must be >= 1")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.width, self.height, self.max_steps = width, height, max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(11,), dtype=np.float32)
        self.snake: deque[tuple[int, int]] = deque()
        self.direction = 0
        self.food: tuple[int, int] | None = None
        self.score = 0
        self.steps = 0
        self.finished = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        x, y = self.width // 2, self.height // 2
        self.snake = deque([(x, y), (x - 1, y), (x - 2, y)])
        self.direction, self.score, self.steps, self.finished = 0, 0, 0, False
        forced = (options or {}).get("food")
        if forced is not None:
            if len(forced) != 2:
                raise ValueError("food must have two coordinates")
            food = (int(forced[0]), int(forced[1]))
            if not self._inside(food) or food in self.snake:
                raise ValueError("food must be a free in-bounds cell")
            self.food = food
        else:
            self.food = self._spawn_food()
        return self._observation(), self._info()

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _spawn_food(self) -> tuple[int, int] | None:
        occupied = set(self.snake)
        available = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied]
        if not available:
            return None
        return available[int(self.np_random.integers(len(available)))]

    def _collision(self, cell: tuple[int, int], *, eating: bool) -> bool:
        body = list(self.snake) if eating else list(self.snake)[:-1]
        return not self._inside(cell) or cell in body

    def _observation(self) -> np.ndarray:
        head = self.snake[0]
        dangers = []
        for turn in (0, 1, -1):
            dx, dy = DIRECTIONS[(self.direction + turn) % 4]
            next_cell = (head[0] + dx, head[1] + dy)
            dangers.append(float(self._collision(next_cell, eating=(next_cell == self.food))))
        directions = [float(self.direction == i) for i in range(4)]
        if self.food is None:
            food_relative = [0.0] * 4
        else:
            fx, fy = self.food
            food_relative = [float(fx > head[0]), float(fx < head[0]), float(fy > head[1]), float(fy < head[1])]
        return np.asarray(dangers + directions + food_relative, dtype=np.float32)

    def _info(self) -> dict[str, int]:
        return {"score": self.score, "length": len(self.snake), "steps": self.steps}

    def step(self, action: int):
        if self.finished:
            raise RuntimeError("episode ended: call reset before step")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        self.direction = (self.direction + (0 if int(action) == 0 else (1 if int(action) == 1 else -1))) % 4
        dx, dy = DIRECTIONS[self.direction]
        head = self.snake[0]
        next_cell = (head[0] + dx, head[1] + dy)
        eating = next_cell == self.food
        self.steps += 1
        if self._collision(next_cell, eating=eating):
            self.finished = True
            return self._observation(), -1.0, True, False, self._info()
        self.snake.appendleft(next_cell)
        if eating:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        won = self.food is None
        truncated = self.steps >= self.max_steps and not won
        self.finished = won or truncated
        reward = 1.0 if eating else -0.01
        return self._observation(), reward, won, truncated, self._info()

    def render(self) -> str:
        board = [["." for _ in range(self.width)] for _ in range(self.height)]
        if self.food:
            board[self.food[1]][self.food[0]] = "F"
        for x, y in self.snake:
            board[y][x] = "o"
        x, y = self.snake[0]
        board[y][x] = "H"
        return "\n".join("".join(row) for row in board)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
