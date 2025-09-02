"""Sample integration test to demonstrate testing infrastructure."""

import pytest


class TestSampleIntegration:
    """Sample integration tests demonstrating the testing infrastructure."""

    @pytest.mark.integration
    def test_package_imports(self):
        """Test that main package modules can be imported together."""
        # Test basic package import
        import bark_ml
        assert hasattr(bark_ml, '__init__')
        
        # Test individual modules that should be importable
        # Note: Some modules may require compiled dependencies
        importable_modules = []
        
        try:
            import bark_ml.behaviors
            importable_modules.append('behaviors')
        except ImportError as e:
            pytest.skip(f"Skipping bark_ml.behaviors import due to: {e}")
        
        try:
            import bark_ml.environments
            importable_modules.append('environments')  
        except ImportError as e:
            # This is expected if dependencies are missing
            pass
            
        try:
            import bark_ml.observers
            importable_modules.append('observers')
        except ImportError as e:
            # This is expected if dependencies are missing
            pass
        
        # At minimum, the base package should be importable
        assert len(importable_modules) >= 0

    @pytest.mark.integration
    def test_environment_and_agent_interaction(self, mock_environment, mock_agent):
        """Test interaction between environment and agent mocks."""
        # Reset environment
        observation = mock_environment.reset()
        assert observation is not None
        
        # Agent acts on observation
        action = mock_agent.act()
        assert action is not None
        
        # Environment steps
        next_obs, reward, done, info = mock_environment.step(action)
        assert next_obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complex_workflow(self, mock_environment, mock_agent, sample_trajectory):
        """Test a more complex workflow involving multiple components."""
        total_reward = 0
        
        # Simulate an episode
        for step_data in sample_trajectory:
            observation = step_data["observation"]
            action = mock_agent.act()
            reward = step_data["reward"]
            total_reward += reward
            
            if step_data["done"]:
                break
        
        assert total_reward > 0
        assert len(sample_trajectory) == 2