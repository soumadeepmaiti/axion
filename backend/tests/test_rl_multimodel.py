"""
Test cases for RL (DQN, PPO) and Multi-Model Ensemble features
Tests backend API endpoints and configuration handling
"""
import pytest
import requests
import os
import sys

# Add backend to path for imports
sys.path.insert(0, '/app/backend')

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://cryptfuse.preview.emergentagent.com').rstrip('/')


class TestBackendImports:
    """Test that all RL and Multi-Model imports work correctly"""
    
    def test_rl_models_import(self):
        """Test RL models can be imported"""
        from ml_models.rl_models import RLTrainer, DQNAgent, PPOAgent, build_rl_model
        assert RLTrainer is not None
        assert DQNAgent is not None
        assert PPOAgent is not None
        assert build_rl_model is not None
    
    def test_multi_model_ensemble_import(self):
        """Test Multi-Model Ensemble can be imported"""
        from ml_models.multi_model_ensemble import MultiModelEnsemble, ModelComparison
        assert MultiModelEnsemble is not None
        assert ModelComparison is not None
    
    def test_advanced_models_helpers_import(self):
        """Test advanced_models helper functions can be imported"""
        from ml_models.advanced_models import build_rl_agent, get_multi_model_ensemble
        assert build_rl_agent is not None
        assert get_multi_model_ensemble is not None
    
    def test_advanced_training_service_import(self):
        """Test advanced training service can be imported"""
        from services.advanced_training_service import AdvancedTrainingService, advanced_training_service
        assert AdvancedTrainingService is not None
        assert advanced_training_service is not None


class TestTrainingStatusAPI:
    """Test training status API endpoint"""
    
    def test_get_advanced_training_status(self):
        """Test GET /api/training/advanced/status returns valid response"""
        response = requests.get(f"{BASE_URL}/api/training/advanced/status")
        assert response.status_code == 200
        
        data = response.json()
        # Verify response structure
        assert 'is_training' in data
        assert 'current_epoch' in data
        assert 'total_epochs' in data
        assert 'network_type' in data
        assert 'error' in data
        
        # Verify data types
        assert isinstance(data['is_training'], bool)
        assert isinstance(data['current_epoch'], int)
        assert isinstance(data['total_epochs'], int)


class TestRLModelsStructure:
    """Test RL model classes have correct structure"""
    
    def test_dqn_agent_structure(self):
        """Test DQNAgent has required methods"""
        from ml_models.rl_models import DQNAgent
        import numpy as np
        
        # Create a small test agent
        state_shape = (10, 5)  # Small shape for testing
        agent = DQNAgent(state_shape, num_actions=3)
        
        # Verify methods exist
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'store_experience')
        assert hasattr(agent, 'train_step')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')
        
        # Verify attributes
        assert agent.num_actions == 3
        assert agent.gamma == 0.99  # Default value
    
    def test_ppo_agent_structure(self):
        """Test PPOAgent has required methods"""
        from ml_models.rl_models import PPOAgent
        
        state_shape = (10, 5)
        agent = PPOAgent(state_shape, num_actions=3)
        
        # Verify methods exist
        assert hasattr(agent, 'select_action')
        assert hasattr(agent, 'compute_advantages')
        assert hasattr(agent, 'train_step')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')
        
        # Verify attributes
        assert agent.num_actions == 3
        assert agent.gamma == 0.99
    
    def test_rl_trainer_structure(self):
        """Test RLTrainer has required methods"""
        from ml_models.rl_models import RLTrainer
        
        trainer = RLTrainer(algorithm='dqn')
        
        # Verify methods exist
        assert hasattr(trainer, 'setup')
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'predict')
        assert hasattr(trainer, 'save')
        assert hasattr(trainer, 'load')
        
        # Verify attributes
        assert trainer.algorithm == 'dqn'


class TestMultiModelEnsembleStructure:
    """Test Multi-Model Ensemble class structure"""
    
    def test_multi_model_ensemble_structure(self):
        """Test MultiModelEnsemble has required methods"""
        from ml_models.multi_model_ensemble import MultiModelEnsemble
        
        input_shape = (50, 10)
        config = {'models': ['lstm', 'gru'], 'ensemble_method': 'weighted'}
        ensemble = MultiModelEnsemble(input_shape, config)
        
        # Verify methods exist
        assert hasattr(ensemble, 'build_models')
        assert hasattr(ensemble, 'fit')
        assert hasattr(ensemble, 'predict')
        assert hasattr(ensemble, 'predict_with_breakdown')
        assert hasattr(ensemble, 'save')
        assert hasattr(ensemble, 'load')
        
        # Verify attributes
        assert ensemble.ensemble_method == 'weighted'
        assert 'lstm' in ensemble.selected_models
        assert 'gru' in ensemble.selected_models
    
    def test_supported_models(self):
        """Test supported model types"""
        from ml_models.multi_model_ensemble import MultiModelEnsemble
        
        assert 'lstm' in MultiModelEnsemble.SUPPORTED_MODELS
        assert 'gru' in MultiModelEnsemble.SUPPORTED_MODELS
        assert 'transformer' in MultiModelEnsemble.SUPPORTED_MODELS
        assert 'cnn_lstm' in MultiModelEnsemble.SUPPORTED_MODELS
    
    def test_ensemble_methods(self):
        """Test supported ensemble methods"""
        from ml_models.multi_model_ensemble import MultiModelEnsemble
        
        assert 'voting' in MultiModelEnsemble.ENSEMBLE_METHODS
        assert 'weighted' in MultiModelEnsemble.ENSEMBLE_METHODS
        assert 'stacking' in MultiModelEnsemble.ENSEMBLE_METHODS
        assert 'blending' in MultiModelEnsemble.ENSEMBLE_METHODS


class TestLLMChatStillWorks:
    """Test that AI Advisor chat still works after code changes"""
    
    def test_llm_providers_endpoint(self):
        """Test GET /api/llm/providers returns providers"""
        response = requests.get(f"{BASE_URL}/api/llm/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert 'providers' in data
        assert len(data['providers']) >= 1
    
    def test_llm_chat_openai(self):
        """Test POST /api/llm/chat with OpenAI provider"""
        response = requests.post(
            f"{BASE_URL}/api/llm/chat",
            json={
                "message": "Say hello in one word",
                "provider": "openai",
                "symbol": "BTC/USDT"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 'content' in data
        assert data['provider'] == 'openai'
        assert len(data['content']) > 0


class TestHealthEndpoint:
    """Test health endpoint still works"""
    
    def test_health_check(self):
        """Test GET /api/health returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'services' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
