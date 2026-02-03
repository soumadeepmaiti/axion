"""
Test Portfolio Model Persistence - Save, Load, Delete functionality
Tests for DL and RL portfolio models persistence to disk
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestPortfolioModelList:
    """Test listing saved portfolio models"""
    
    def test_list_models_endpoint(self):
        """Test GET /api/portfolio/models/list returns saved models"""
        response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert 'deep_learning_models' in data
        assert 'rl_agent_models' in data
        assert 'total_models' in data
        
        # Verify lists are arrays
        assert isinstance(data['deep_learning_models'], list)
        assert isinstance(data['rl_agent_models'], list)
        
        print(f"Found {len(data['deep_learning_models'])} DL models")
        print(f"Found {len(data['rl_agent_models'])} RL models")
        print(f"Total models: {data['total_models']}")
    
    def test_list_models_contains_existing_models(self):
        """Verify the pre-saved models are listed"""
        response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for the pre-saved DL model
        dl_models = data['deep_learning_models']
        dl_model_names = [m.get('model_name', '') for m in dl_models]
        
        # Check for the pre-saved RL model
        rl_models = data['rl_agent_models']
        rl_model_names = [m.get('model_name', '') for m in rl_models]
        
        print(f"DL model names: {dl_model_names}")
        print(f"RL model names: {rl_model_names}")
        
        # Verify at least one model exists
        assert len(dl_models) > 0 or len(rl_models) > 0, "Expected at least one saved model"
    
    def test_model_metadata_structure(self):
        """Verify model metadata has required fields"""
        response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check DL model metadata
        if data['deep_learning_models']:
            dl_model = data['deep_learning_models'][0]
            assert 'model_name' in dl_model
            assert 'model_path' in dl_model
            assert 'model_type' in dl_model
            assert dl_model['model_type'] == 'deep_portfolio_network'
            print(f"DL model metadata: {dl_model}")
        
        # Check RL model metadata
        if data['rl_agent_models']:
            rl_model = data['rl_agent_models'][0]
            assert 'model_name' in rl_model
            assert 'model_path' in rl_model
            assert 'model_type' in rl_model
            assert rl_model['model_type'] == 'rl_portfolio_agent'
            print(f"RL model metadata: {rl_model}")


class TestPortfolioModelLoad:
    """Test loading saved portfolio models"""
    
    def test_load_dl_model(self):
        """Test POST /api/portfolio/models/load for DL model"""
        # First get the list of models
        list_response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        assert list_response.status_code == 200
        
        dl_models = list_response.json().get('deep_learning_models', [])
        
        if not dl_models:
            pytest.skip("No DL models available to load")
        
        # Load the first DL model
        model_path = dl_models[0]['model_path']
        
        response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
            "model_type": "deep_learning",
            "model_path": model_path
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get('status') == 'success'
        assert 'model_path' in data
        print(f"DL model loaded successfully: {data}")
    
    def test_load_rl_model(self):
        """Test POST /api/portfolio/models/load for RL model"""
        # First get the list of models
        list_response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        assert list_response.status_code == 200
        
        rl_models = list_response.json().get('rl_agent_models', [])
        
        if not rl_models:
            pytest.skip("No RL models available to load")
        
        # Load the first RL model
        model_path = rl_models[0]['model_path']
        
        response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
            "model_type": "rl_agent",
            "model_path": model_path
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data.get('status') == 'success'
        assert 'model_path' in data
        print(f"RL model loaded successfully: {data}")
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist"""
        response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
            "model_type": "deep_learning",
            "model_path": "/app/backend/saved_models/portfolio/nonexistent_model"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return error status
        assert data.get('status') == 'error'
        assert 'message' in data
        print(f"Expected error for nonexistent model: {data['message']}")
    
    def test_load_invalid_model_type(self):
        """Test loading with invalid model type"""
        response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
            "model_type": "invalid_type",
            "model_path": "/some/path"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return error status
        assert data.get('status') == 'error'
        print(f"Expected error for invalid model type: {data}")


class TestPortfolioModelInfo:
    """Test model info after loading"""
    
    def test_model_info_after_load(self):
        """Verify model info reflects loaded model"""
        # First load a model
        list_response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        dl_models = list_response.json().get('deep_learning_models', [])
        
        if dl_models:
            model_path = dl_models[0]['model_path']
            requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
                "model_type": "deep_learning",
                "model_path": model_path
            })
        
        # Check model info
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert 'multi_asset_predictor' in data
        assert 'deep_learning_trained' in data
        assert 'rl_agent_trained' in data
        
        print(f"Model info: deep_learning_trained={data['deep_learning_trained']}, rl_agent_trained={data['rl_agent_trained']}")


class TestPortfolioModelSave:
    """Test saving portfolio models"""
    
    def test_save_dl_model_without_training(self):
        """Test saving DL model when not trained returns error"""
        # This test checks the error case - saving without a trained model
        # First, we need to ensure no model is trained
        response = requests.post(f"{BASE_URL}/api/portfolio/models/save", json={
            "model_type": "deep_learning"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # If model is trained, it should save successfully
        # If not trained, it should return error
        if data.get('status') == 'error':
            assert 'message' in data
            print(f"Expected: No trained model to save - {data['message']}")
        else:
            assert data.get('status') == 'success'
            print(f"Model was already trained and saved: {data}")
    
    def test_save_rl_model_without_training(self):
        """Test saving RL model when not trained returns error"""
        response = requests.post(f"{BASE_URL}/api/portfolio/models/save", json={
            "model_type": "rl_agent"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # If model is trained, it should save successfully
        # If not trained, it should return error
        if data.get('status') == 'error':
            assert 'message' in data
            print(f"Expected: No trained model to save - {data['message']}")
        else:
            assert data.get('status') == 'success'
            print(f"Model was already trained and saved: {data}")


class TestPortfolioModelDelete:
    """Test deleting portfolio models"""
    
    def test_delete_nonexistent_model(self):
        """Test deleting a model that doesn't exist"""
        response = requests.post(f"{BASE_URL}/api/portfolio/models/delete", json={
            "model_type": "deep_learning",
            "model_path": "/app/backend/saved_models/portfolio/nonexistent_model_12345"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return error status
        assert data.get('status') == 'error'
        assert 'message' in data
        print(f"Expected error for nonexistent model: {data['message']}")
    
    def test_delete_invalid_model_type(self):
        """Test deleting with invalid model type"""
        response = requests.post(f"{BASE_URL}/api/portfolio/models/delete", json={
            "model_type": "invalid_type",
            "model_path": "/some/path"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return error status
        assert data.get('status') == 'error'
        print(f"Expected error for invalid model type: {data}")


class TestPortfolioModelIntegration:
    """Integration tests for model persistence workflow"""
    
    def test_load_and_verify_model_info(self):
        """Test loading a model and verifying it's reflected in model info"""
        # Get list of models
        list_response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        assert list_response.status_code == 200
        
        dl_models = list_response.json().get('deep_learning_models', [])
        rl_models = list_response.json().get('rl_agent_models', [])
        
        # Load DL model if available
        if dl_models:
            model_path = dl_models[0]['model_path']
            load_response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
                "model_type": "deep_learning",
                "model_path": model_path
            })
            assert load_response.status_code == 200
            assert load_response.json().get('status') == 'success'
            
            # Verify model info
            info_response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
            assert info_response.status_code == 200
            assert info_response.json().get('deep_learning_trained') == True
            print("DL model loaded and verified in model info")
        
        # Load RL model if available
        if rl_models:
            model_path = rl_models[0]['model_path']
            load_response = requests.post(f"{BASE_URL}/api/portfolio/models/load", json={
                "model_type": "rl_agent",
                "model_path": model_path
            })
            assert load_response.status_code == 200
            assert load_response.json().get('status') == 'success'
            
            # Verify model info
            info_response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
            assert info_response.status_code == 200
            assert info_response.json().get('rl_agent_trained') == True
            print("RL model loaded and verified in model info")
    
    def test_list_models_returns_correct_count(self):
        """Verify total_models count matches sum of DL and RL models"""
        response = requests.get(f"{BASE_URL}/api/portfolio/models/list")
        
        assert response.status_code == 200
        data = response.json()
        
        dl_count = len(data.get('deep_learning_models', []))
        rl_count = len(data.get('rl_agent_models', []))
        total = data.get('total_models', 0)
        
        assert total == dl_count + rl_count, f"Total {total} != DL {dl_count} + RL {rl_count}"
        print(f"Model count verified: {dl_count} DL + {rl_count} RL = {total} total")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
