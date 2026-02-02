"""
Portfolio Optimization Phase 2 API Tests
Tests for Deep Learning, RL Agent, and Hybrid strategies:
- /api/portfolio/train-model - Train DL and RL models
- /api/portfolio/optimize - Test all 4 strategies
- /api/portfolio/model-info - Verify model training status
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestPortfolioModelInfo:
    """Test /api/portfolio/model-info endpoint for Phase 2 models"""
    
    def test_model_info_returns_dl_status(self):
        """Test that model-info returns deep_learning_trained status"""
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "deep_learning_trained" in data
        assert isinstance(data["deep_learning_trained"], bool)
        
    def test_model_info_returns_rl_status(self):
        """Test that model-info returns rl_agent_trained status"""
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        assert response.status_code == 200
        
        data = response.json()
        assert "rl_agent_trained" in data
        assert isinstance(data["rl_agent_trained"], bool)
        
    def test_model_info_dl_details_when_trained(self):
        """Test that model-info returns DL model details when trained"""
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        assert response.status_code == 200
        
        data = response.json()
        if data.get("deep_learning_trained"):
            assert "deep_learning_info" in data
            dl_info = data["deep_learning_info"]
            assert "is_trained" in dl_info
            assert "n_assets" in dl_info
            assert "asset_names" in dl_info
            assert "model_params" in dl_info
            assert dl_info["is_trained"] == True
            
    def test_model_info_rl_details_when_trained(self):
        """Test that model-info returns RL agent details when trained"""
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        assert response.status_code == 200
        
        data = response.json()
        if data.get("rl_agent_trained"):
            assert "rl_agent_info" in data
            rl_info = data["rl_agent_info"]
            assert "is_trained" in rl_info
            assert "n_assets" in rl_info
            assert "asset_names" in rl_info
            assert "training_result" in rl_info
            assert rl_info["is_trained"] == True


class TestPortfolioTrainModel:
    """Test /api/portfolio/train-model endpoint"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before training tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_train_model_endpoint_exists(self):
        """Test that train-model endpoint exists and accepts requests"""
        payload = {
            "model_type": "deep_learning",
            "epochs": 10
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/train-model", json=payload)
        # Should return 200 (started) or error if already trained
        assert response.status_code == 200
        
    def test_train_dl_model_returns_started(self):
        """Test that training DL model returns started status"""
        payload = {
            "model_type": "deep_learning",
            "epochs": 10
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/train-model", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        # Status should be 'started' or 'error' if no data
        assert data["status"] in ["started", "error"]
        
    def test_train_rl_agent_returns_started(self):
        """Test that training RL agent returns started status"""
        payload = {
            "model_type": "rl_agent",
            "n_episodes": 10
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/train-model", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["started", "error"]
        
    def test_train_invalid_model_type(self):
        """Test that invalid model type returns error"""
        payload = {
            "model_type": "invalid_model",
            "epochs": 10
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/train-model", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "error"


class TestPortfolioOptimizePhase2:
    """Test /api/portfolio/optimize with Phase 2 strategies"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before optimization tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_optimize_deep_learning_strategy(self):
        """Test portfolio optimization with deep_learning strategy"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "deep_learning",
            "objective": "max_sharpe",
            "horizon": "7d",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["strategy"] == "deep_learning"
        
        # If trained, should return success with allocations
        if data["status"] == "success":
            assert "allocations" in data
            assert "metrics" in data
            assert len(data["allocations"]) > 0
        # If not trained, should return not_trained status
        elif data["status"] == "not_trained":
            assert "message" in data
            
    def test_optimize_rl_agent_strategy(self):
        """Test portfolio optimization with rl_agent strategy"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "rl_agent",
            "objective": "max_sharpe",
            "horizon": "7d",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["strategy"] == "rl_agent"
        
        # If trained, should return success with allocations
        if data["status"] == "success":
            assert "allocations" in data
            assert "metrics" in data
            assert len(data["allocations"]) > 0
        # If not trained, should return not_trained status
        elif data["status"] == "not_trained":
            assert "message" in data
            
    def test_optimize_hybrid_strategy(self):
        """Test portfolio optimization with hybrid strategy"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "hybrid",
            "objective": "max_sharpe",
            "horizon": "7d",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["strategy"] == "hybrid"
        
        # Hybrid should always work (at least with traditional_ml)
        if data["status"] == "success":
            assert "allocations" in data
            assert "metrics" in data
            # Should indicate which strategies were combined
            if "strategies_combined" in data:
                assert "traditional_ml" in data["strategies_combined"]
                
    def test_optimize_compare_all_returns_4_strategies(self):
        """Test that compare_all=True returns all 4 strategies"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "horizon": "7d",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["type"] == "comparison"
        assert "strategies" in data
        
        strategies = data["strategies"]
        # All 4 strategies should be present
        assert "traditional_ml" in strategies
        assert "deep_learning" in strategies
        assert "rl_agent" in strategies
        assert "hybrid" in strategies
        
    def test_compare_all_traditional_ml_success(self):
        """Test that traditional_ml always succeeds in compare_all"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        strategies = data["strategies"]
        
        # Traditional ML should always succeed
        assert strategies["traditional_ml"]["status"] == "success"
        assert "allocations" in strategies["traditional_ml"]
        assert "metrics" in strategies["traditional_ml"]
        
    def test_compare_all_dl_status_correct(self):
        """Test that deep_learning status is correct based on training"""
        # First check if DL is trained
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        dl_trained = model_info.get("deep_learning_trained", False)
        
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        dl_result = data["strategies"]["deep_learning"]
        
        if dl_trained:
            assert dl_result["status"] == "success"
            assert "allocations" in dl_result
        else:
            assert dl_result["status"] == "not_trained"
            
    def test_compare_all_rl_status_correct(self):
        """Test that rl_agent status is correct based on training"""
        # First check if RL is trained
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        rl_trained = model_info.get("rl_agent_trained", False)
        
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        rl_result = data["strategies"]["rl_agent"]
        
        if rl_trained:
            assert rl_result["status"] == "success"
            assert "allocations" in rl_result
        else:
            assert rl_result["status"] == "not_trained"
            
    def test_compare_all_returns_recommended_strategy(self):
        """Test that compare_all returns a recommended strategy"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        assert "recommended" in data
        # Recommended should be one of the 4 strategies
        assert data["recommended"] in ["traditional_ml", "deep_learning", "rl_agent", "hybrid"]
        
    def test_compare_all_returns_training_status(self):
        """Test that compare_all returns model training status"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": True
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        assert "deep_learning_trained" in data
        assert "rl_agent_trained" in data


class TestPhase2AllocationStructure:
    """Test allocation structure for Phase 2 strategies"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_dl_allocations_sum_to_100(self):
        """Test that DL allocations sum to approximately 100%"""
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        if not model_info.get("deep_learning_trained"):
            pytest.skip("Deep Learning model not trained")
            
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "deep_learning",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            total_weight = sum(a["weight"] for a in data["allocations"])
            assert 99 <= total_weight <= 101  # Allow small rounding errors
            
    def test_rl_allocations_sum_to_100(self):
        """Test that RL allocations sum to approximately 100%"""
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        if not model_info.get("rl_agent_trained"):
            pytest.skip("RL Agent not trained")
            
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "rl_agent",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            total_weight = sum(a["weight"] for a in data["allocations"])
            assert 99 <= total_weight <= 101
            
    def test_hybrid_allocations_sum_to_100(self):
        """Test that Hybrid allocations sum to approximately 100%"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "hybrid",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            total_weight = sum(a["weight"] for a in data["allocations"])
            assert 99 <= total_weight <= 101
            
    def test_dl_metrics_structure(self):
        """Test that DL strategy returns proper metrics"""
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        if not model_info.get("deep_learning_trained"):
            pytest.skip("Deep Learning model not trained")
            
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "deep_learning",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            assert "metrics" in data
            metrics = data["metrics"]
            assert "expected_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics
            assert "num_assets" in metrics
            
    def test_rl_metrics_structure(self):
        """Test that RL strategy returns proper metrics"""
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        if not model_info.get("rl_agent_trained"):
            pytest.skip("RL Agent not trained")
            
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "rl_agent",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            assert "metrics" in data
            metrics = data["metrics"]
            assert "expected_return" in metrics
            assert "volatility" in metrics
            assert "sharpe_ratio" in metrics


class TestHybridStrategyDetails:
    """Test Hybrid strategy specific features"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_hybrid_includes_strategies_combined(self):
        """Test that hybrid strategy indicates which strategies were combined"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "hybrid",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            assert "strategies_combined" in data
            # Should always include traditional_ml
            assert "traditional_ml" in data["strategies_combined"]
            
    def test_hybrid_combines_trained_models(self):
        """Test that hybrid combines all trained models"""
        model_info = requests.get(f"{BASE_URL}/api/portfolio/model-info").json()
        dl_trained = model_info.get("deep_learning_trained", False)
        rl_trained = model_info.get("rl_agent_trained", False)
        
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "hybrid",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        data = response.json()
        
        if data["status"] == "success":
            strategies_combined = data.get("strategies_combined", [])
            
            if dl_trained:
                assert "deep_learning" in strategies_combined
            if rl_trained:
                assert "rl_agent" in strategies_combined


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
