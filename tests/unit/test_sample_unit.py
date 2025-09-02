"""Sample unit test to demonstrate testing infrastructure."""

import pytest
import numpy as np


class TestSampleUnit:
    """Sample unit tests demonstrating the testing infrastructure."""

    @pytest.mark.unit
    def test_basic_assertion(self):
        """Test basic assertion functionality."""
        assert 2 + 2 == 4

    @pytest.mark.unit
    def test_numpy_operations(self):
        """Test numpy operations for ML components."""
        arr = np.array([1, 2, 3, 4])
        assert arr.sum() == 10
        assert arr.mean() == 2.5

    @pytest.mark.unit
    def test_with_fixture(self, sample_config):
        """Test using a shared fixture."""
        assert sample_config["environment"]["max_steps"] == 100
        assert sample_config["agent"]["learning_rate"] == 0.001

    @pytest.mark.unit
    def test_mock_usage(self, mock_model):
        """Test using a mock fixture."""
        prediction = mock_model.predict([[1, 2, 3]])
        assert prediction is not None
        mock_model.predict.assert_called_once_with([[1, 2, 3]])

    @pytest.mark.unit
    def test_temporary_directory(self, temp_dir):
        """Test using temporary directory fixture."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"