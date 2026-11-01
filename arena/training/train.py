"""Reproducible Snake DQN training CLI: python -m arena.training.train."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import platform
import random
import numpy as np
import torch
import gymnasium
from arena.agents.dqn import DQNAgent, DQNConfig
from arena.envs.snake import SnakeEnv
from arena.evaluation.benchmark import compare

def train(*, episodes: int = 100, eval_episodes: int = 20, width: int = 10,
          height: int = 10, max_steps: int = 200, seed: int = 42,
          out: str | Path = "experiments/local") -> dict:
    if episodes < 1 or eval_episodes < 1:
        raise ValueError("episodes and eval_episodes must be positive")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    config = DQNConfig()
    agent = DQNAgent(config, seed=seed)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    with SnakeEnv(width=width, height=height, max_steps=max_steps) as env:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward, losses = 0.0, []
            while True:
                action = agent.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                agent.remember(observation, action, reward, next_observation, terminated)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                total_reward += reward
                observation = next_observation
                if terminated or truncated:
                    history.append({"episode": episode + 1, "seed": seed + episode,
                                    "score": info["score"], "steps": info["steps"],
                                    "reward": round(total_reward, 6),
                                    "loss": round(sum(losses) / len(losses), 6) if losses else "",
                                    "epsilon": round(agent.epsilon(), 6)})
                    break
    agent.save(output_dir / "model.pt")
    eval_seeds = list(range(seed + 100_000, seed + 100_000 + eval_episodes))
    metrics = compare(agent, seeds=eval_seeds, width=width, height=height, max_steps=max_steps)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    metadata = {"seed": seed, "episodes": episodes, "eval_episodes": eval_episodes,
                "width": width, "height": height, "max_steps": max_steps,
                "training_seeds": [seed, seed + episodes - 1], "evaluation_seeds": eval_seeds,
                "config": asdict(config), "versions": {"python": platform.python_version(),
                  "numpy": np.__version__, "torch": torch.__version__, "gymnasium": gymnasium.__version__},
                "gradient_updates": agent.updates,
                "determinism": {"algorithms": torch.are_deterministic_algorithms_enabled(),
                                 "torch_threads": torch.get_num_threads()}}
    with (output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return {"output": str(output_dir), "gradient_updates": agent.updates,
            "trained_score_mean": metrics["trained"]["summary"]["score"]["mean"],
            "random_score_mean": metrics["random"]["summary"]["score"]["mean"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="experiments/local")
    args = parser.parse_args()
    print(json.dumps(train(**vars(args)), indent=2))

if __name__ == "__main__":
    main()
