"""Paired-seed evaluation of trained agent versus random policy."""
from __future__ import annotations
from statistics import mean, median, stdev
from typing import Callable
import random
from arena.envs.snake import SnakeEnv

def run_episode(env: SnakeEnv, policy: Callable, seed: int) -> dict[str, float | int]:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {"score": info["score"], "steps": info["steps"], "reward": round(total_reward, 6)}

def _summary(results: list[dict[str, float | int]]) -> dict:
    summary = {}
    for key in ("score", "steps", "reward"):
        values = [float(result[key]) for result in results]
        summary[key] = {"mean": mean(values), "median": median(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "min": min(values), "max": max(values)}
    return summary

def compare(agent, *, seeds: list[int], width: int, height: int, max_steps: int) -> dict:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation needs nonempty distinct seeds")
    trained, random_baseline = [], []
    with SnakeEnv(width=width, height=height, max_steps=max_steps) as env:
        for seed in seeds:
            trained.append(run_episode(env, lambda obs: agent.act(obs, greedy=True), seed))
            rng = random.Random(seed + 50_000_000)
            random_baseline.append(run_episode(env, lambda _obs: rng.randrange(env.action_space.n), seed))
    return {"seeds": seeds, "trained": {"summary": _summary(trained), "episodes": trained},
            "random": {"summary": _summary(random_baseline), "episodes": random_baseline}}
