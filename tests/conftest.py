"""Shared pytest fixtures for BARK-ML testing infrastructure."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, List
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after test completion."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file in a temporary directory."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "environment": {
            "name": "test_env",
            "max_steps": 100,
            "observation_space": "simple",
            "action_space": "continuous"
        },
        "agent": {
            "type": "test_agent",
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "training": {
            "episodes": 1000,
            "log_interval": 10,
            "save_interval": 100
        }
    }


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing agents."""
    mock_env = Mock()
    mock_env.observation_space = Mock()
    mock_env.action_space = Mock()
    mock_env.reset.return_value = np.array([0.0, 0.0, 1.0])
    mock_env.step.return_value = (
        np.array([0.1, 0.1, 0.9]),  # observation
        1.0,  # reward
        False,  # done
        {}  # info
    )
    return mock_env


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing environments."""
    mock_agent = Mock()
    mock_agent.act.return_value = np.array([0.5])
    mock_agent.train.return_value = {"loss": 0.1, "accuracy": 0.95}
    return mock_agent


@pytest.fixture
def sample_trajectory() -> List[Dict[str, Any]]:
    """Provide a sample trajectory for testing."""
    return [
        {
            "observation": np.array([0.0, 0.0, 1.0]),
            "action": np.array([0.5]),
            "reward": 1.0,
            "done": False,
            "next_observation": np.array([0.1, 0.1, 0.9])
        },
        {
            "observation": np.array([0.1, 0.1, 0.9]),
            "action": np.array([0.3]),
            "reward": 0.8,
            "done": True,
            "next_observation": np.array([0.0, 0.0, 0.0])
        }
    ]


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.9]])
    mock_model.fit.return_value = {"loss": [0.5, 0.3, 0.1]}
    mock_model.save.return_value = None
    mock_model.load.return_value = None
    return mock_model


@pytest.fixture
def sample_network_weights() -> Dict[str, np.ndarray]:
    """Provide sample neural network weights for testing."""
    return {
        "layer1": np.random.randn(10, 5),
        "layer2": np.random.randn(5, 3),
        "bias1": np.random.randn(5),
        "bias2": np.random.randn(3)
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    mock_logger.debug = Mock()
    return mock_logger


@pytest.fixture
def clean_environment():
    """Clean environment variables before and after tests."""
    original_env = dict(os.environ)
    # Clean test-related environment variables
    test_vars = [var for var in os.environ.keys() if var.startswith('TEST_')]
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    # If using other random libraries, reset their seeds here too


@pytest.fixture
def mock_gym_env():
    """Create a mock gym environment with standard interface."""
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (1,)
    mock_env.action_space.low = np.array([-1.0])
    mock_env.action_space.high = np.array([1.0])
    mock_env.spec = Mock()
    mock_env.spec.id = "TestEnv-v0"
    return mock_env


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Provide sample metrics for testing evaluators."""
    return {
        "reward_mean": 15.5,
        "reward_std": 2.3,
        "episode_length_mean": 89.2,
        "episode_length_std": 12.1,
        "success_rate": 0.85,
        "collision_rate": 0.05
    }