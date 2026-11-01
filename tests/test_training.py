import csv
import json
import numpy as np
import pytest
import torch
from arena.agents.dqn import DQNAgent, DQNConfig
from arena.evaluation.benchmark import compare
from arena.training.train import train

def test_dqn_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(9)
    agent = DQNAgent(DQNConfig(batch_size=2, warmup=2), seed=9)
    observation = np.zeros(11, dtype=np.float32)
    agent.remember(observation, 0, 1.0, observation, False)
    agent.remember(observation, 2, -1.0, observation, True)
    assert agent.learn() is not None
    path = tmp_path / "model.pt"
    agent.save(path)
    restored = DQNAgent.load(path)
    assert restored.steps == agent.steps and restored.updates == agent.updates
    assert restored.act(observation, greedy=True) == agent.act(observation, greedy=True)
    for key, weight in agent.online.state_dict().items():
        assert torch.equal(weight, restored.online.state_dict()[key])

def test_benchmark_paired_seeds():
    agent = DQNAgent(DQNConfig(), seed=42)
    result = compare(agent, seeds=[500, 501, 502], width=6, height=6, max_steps=30)
    assert result["seeds"] == [500, 501, 502]
    assert len(result["random"]["episodes"]) == 3
    assert result["trained"]["summary"]["steps"]["max"] <= 30
    with pytest.raises(ValueError):
        compare(agent, seeds=[1, 1], width=6, height=6, max_steps=30)

def test_short_training_generates_artifacts(tmp_path):
    output = tmp_path / "trial"
    result = train(episodes=6, eval_episodes=4, width=6, height=6, max_steps=25, seed=18, out=output)
    assert result["output"] == str(output)
    assert (output / "model.pt").is_file()
    assert DQNAgent.load(output / "model.pt").steps > 0
    metadata = json.loads((output / "run.json").read_text())
    report = json.loads((output / "evaluation.json").read_text())
    assert metadata["training_seeds"] == [18, 23]
    assert metadata["evaluation_seeds"] == [100018, 100019, 100020, 100021]
    assert report["seeds"] == metadata["evaluation_seeds"]
    with (output / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
