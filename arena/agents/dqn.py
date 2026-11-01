"""CPU-friendly DQN agent with target network and replay memory."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

@dataclass(frozen=True)
class DQNConfig:
    observation_dim: int = 11
    actions: int = 3
    hidden: int = 64
    gamma: float = 0.98
    learning_rate: float = 0.001
    capacity: int = 20_000
    batch_size: int = 64
    warmup: int = 128
    target_updates: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    def __post_init__(self):
        if self.observation_dim < 1 or self.actions < 1 or self.hidden < 1:
            raise ValueError("invalid network dimensions")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("epsilon must be in [0,1], end <= start")
        if min(self.batch_size, self.warmup, self.capacity, self.target_updates, self.epsilon_decay_steps) < 1:
            raise ValueError("buffer, batch, target and epsilon parameters must be positive")
        if self.capacity < self.batch_size:
            raise ValueError("capacity must accommodate batch_size")

class QNetwork(nn.Module):
    def __init__(self, config: DQNConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQNAgent:
    def __init__(self, config: DQNConfig, seed: int = 42):
        self.config = config
        self.random = random.Random(seed)
        self.online = QNetwork(config)
        self.target = QNetwork(config)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=config.capacity)
        self.steps, self.updates = 0, 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and self.random.random() < self.epsilon():
            return self.random.randrange(self.config.actions)
        x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(self.online(x).argmax(dim=1).item())

    def remember(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, terminated: bool) -> None:
        self.replay.append((observation.copy(), action, reward, next_observation.copy(), terminated))
        self.steps += 1

    def learn(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.warmup):
            return None
        batch = self.random.sample(self.replay, self.config.batch_size)
        observations = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        next_observations = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        terminals = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        prediction = self.online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_value = self.target(next_observations).max(dim=1).values
            target = rewards + self.config.gamma * (1 - terminals) * next_value
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "online": self.online.state_dict(),
                    "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "steps": self.steps, "updates": self.updates}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(DQNConfig(**data["config"]))
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.steps, agent.updates = int(data["steps"]), int(data["updates"])
        return agent
