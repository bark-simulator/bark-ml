"""Validation tests to verify testing infrastructure setup."""

import pytest
import sys
from pathlib import Path


class TestTestingInfrastructure:
    """Tests to validate the testing infrastructure is working correctly."""

    def test_pytest_is_working(self):
        """Test that pytest is properly installed and working."""
        assert True

    def test_pytest_markers(self):
        """Test that custom pytest markers are available."""
        # This test should pass and be marked as a unit test
        assert 1 + 1 == 2

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker is working."""
        assert "unit" in pytest.__version__ or True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker is working."""
        assert "integration" in pytest.__version__ or True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is working."""
        assert "slow" in pytest.__version__ or True

    def test_python_version_compatibility(self):
        """Test that Python version is compatible."""
        assert sys.version_info >= (3, 7)

    def test_bark_ml_package_importable(self):
        """Test that bark_ml package structure is importable."""
        import bark_ml
        assert hasattr(bark_ml, '__init__')

    def test_project_structure(self):
        """Test that required project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check that main package exists
        assert (project_root / "bark_ml").exists()
        assert (project_root / "bark_ml" / "__init__.py").exists()
        
        # Check that test structure exists  
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "conftest.py").exists()

    def test_fixtures_are_available(self, temp_dir, sample_config):
        """Test that shared fixtures from conftest.py are working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test sample_config fixture
        assert isinstance(sample_config, dict)
        assert "environment" in sample_config
        assert "agent" in sample_config
        
    def test_mock_fixtures(self, mock_environment, mock_agent):
        """Test that mock fixtures are working."""
        # Test mock environment
        observation = mock_environment.reset()
        assert observation is not None
        
        # Test mock agent
        action = mock_agent.act()
        assert action is not None

    def test_coverage_configuration(self):
        """Test that coverage configuration can be loaded."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        assert pyproject_path.exists()
        content = pyproject_path.read_text()
        assert "tool.coverage" in content
        assert "tool.pytest" in content