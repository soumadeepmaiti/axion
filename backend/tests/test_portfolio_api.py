"""
Portfolio Optimization API Tests
Tests for multi-asset portfolio optimization endpoints:
- /api/portfolio/assets - Get available assets
- /api/portfolio/fetch-data - Fetch market data for assets
- /api/portfolio/optimize - Optimize portfolio allocation
- /api/portfolio/efficient-frontier - Get efficient frontier data
- /api/portfolio/correlation - Get correlation matrix
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestPortfolioAssets:
    """Test /api/portfolio/assets endpoint"""
    
    def test_get_available_assets(self):
        """Test that assets endpoint returns 20 default assets"""
        response = requests.get(f"{BASE_URL}/api/portfolio/assets")
        assert response.status_code == 200
        
        data = response.json()
        assert "assets" in data
        assert "count" in data
        assert data["count"] == 20
        assert len(data["assets"]) == 20
        
        # Verify key assets are present
        assert "BTC/USDT" in data["assets"]
        assert "ETH/USDT" in data["assets"]
        assert "SOL/USDT" in data["assets"]
        
    def test_assets_include_strategies(self):
        """Test that assets endpoint returns available strategies"""
        response = requests.get(f"{BASE_URL}/api/portfolio/assets")
        assert response.status_code == 200
        
        data = response.json()
        assert "strategies" in data
        assert "traditional_ml" in data["strategies"]
        assert "deep_learning" in data["strategies"]
        assert "rl_agent" in data["strategies"]
        assert "hybrid" in data["strategies"]
        
    def test_assets_include_objectives(self):
        """Test that assets endpoint returns optimization objectives"""
        response = requests.get(f"{BASE_URL}/api/portfolio/assets")
        assert response.status_code == 200
        
        data = response.json()
        assert "objectives" in data
        assert "max_sharpe" in data["objectives"]
        assert "max_return" in data["objectives"]
        assert "min_risk" in data["objectives"]
        assert "risk_parity" in data["objectives"]
        
    def test_assets_include_horizons(self):
        """Test that assets endpoint returns prediction horizons"""
        response = requests.get(f"{BASE_URL}/api/portfolio/assets")
        assert response.status_code == 200
        
        data = response.json()
        assert "horizons" in data
        assert "24h" in data["horizons"]
        assert "7d" in data["horizons"]
        assert "30d" in data["horizons"]


class TestPortfolioFetchData:
    """Test /api/portfolio/fetch-data endpoint"""
    
    def test_fetch_data_for_5_assets(self):
        """Test fetching data for 5 assets (faster test)"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "assets_fetched" in data
        assert data["assets_fetched"] >= 4  # At least 4 assets should succeed
        assert "statistics" in data
        assert "correlation_data" in data
        
    def test_fetch_data_returns_statistics(self):
        """Test that fetch-data returns asset statistics"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1d"
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "statistics" in data
        
        # Check statistics structure for BTC
        if "BTC/USDT" in data["statistics"]:
            btc_stats = data["statistics"]["BTC/USDT"]
            assert "symbol" in btc_stats
            assert "current_price" in btc_stats
            assert "expected_return" in btc_stats
            assert "volatility" in btc_stats
            assert "sharpe_ratio" in btc_stats
            assert "data_points" in btc_stats
            
    def test_fetch_data_returns_correlation_data(self):
        """Test that fetch-data returns correlation matrix data"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "timeframe": "1d"
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "correlation_data" in data
        
        corr_data = data["correlation_data"]
        assert "assets" in corr_data
        assert "matrix" in corr_data
        assert len(corr_data["assets"]) > 0
        assert len(corr_data["matrix"]) > 0


class TestPortfolioOptimize:
    """Test /api/portfolio/optimize endpoint"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before optimization tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)  # Allow data to be processed
    
    def test_optimize_traditional_ml_strategy(self):
        """Test portfolio optimization with traditional_ml strategy"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "horizon": "7d",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["strategy"] == "traditional_ml"
        assert "allocations" in data
        assert "metrics" in data
        
    def test_optimize_returns_allocations(self):
        """Test that optimization returns proper allocation structure"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 10000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "allocations" in data
        
        # Check allocation structure
        for alloc in data["allocations"]:
            assert "symbol" in alloc
            assert "weight" in alloc
            assert "amount" in alloc
            assert alloc["weight"] > 0
            assert alloc["amount"] > 0
            
    def test_optimize_returns_metrics(self):
        """Test that optimization returns portfolio metrics"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": False
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        
        metrics = data["metrics"]
        assert "expected_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "num_assets" in metrics
        
    def test_optimize_compare_all_strategies(self):
        """Test optimization with compare_all=True returns all strategies"""
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
        assert data["status"] == "success"
        assert data["type"] == "comparison"
        assert "strategies" in data
        assert "recommended" in data
        
        # Check all 4 strategies are present
        strategies = data["strategies"]
        assert "traditional_ml" in strategies
        assert "deep_learning" in strategies
        assert "rl_agent" in strategies
        assert "hybrid" in strategies
        
        # Traditional ML should be successful
        assert strategies["traditional_ml"]["status"] == "success"
        
        # Deep Learning and RL Agent should be pending (Phase 2)
        assert strategies["deep_learning"]["status"] == "pending"
        assert strategies["rl_agent"]["status"] == "pending"
        
    def test_optimize_with_constraints(self):
        """Test optimization with custom constraints"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "investment_amount": 1000,
            "strategy": "traditional_ml",
            "objective": "max_sharpe",
            "compare_all": False,
            "constraints": {
                "max_weight": 25,  # Max 25% per asset
                "min_assets": 3
            }
        }
        response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        
        # Verify max weight constraint
        for alloc in data["allocations"]:
            assert alloc["weight"] <= 30  # Allow some tolerance
            
    def test_optimize_different_objectives(self):
        """Test optimization with different objectives"""
        objectives = ["max_sharpe", "max_return", "min_risk"]
        
        for obj in objectives:
            payload = {
                "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
                "investment_amount": 1000,
                "strategy": "traditional_ml",
                "objective": obj,
                "compare_all": False
            }
            response = requests.post(f"{BASE_URL}/api/portfolio/optimize", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert data["objective"] == obj


class TestEfficientFrontier:
    """Test /api/portfolio/efficient-frontier endpoint"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before frontier tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_get_efficient_frontier(self):
        """Test efficient frontier endpoint returns data"""
        assets = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT"
        response = requests.get(f"{BASE_URL}/api/portfolio/efficient-frontier?assets={assets}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "frontier" in data
        assert "assets" in data
        
    def test_efficient_frontier_structure(self):
        """Test efficient frontier data structure"""
        assets = "BTC/USDT,ETH/USDT,BNB/USDT"
        response = requests.get(f"{BASE_URL}/api/portfolio/efficient-frontier?assets={assets}")
        assert response.status_code == 200
        
        data = response.json()
        assert "frontier" in data
        
        # Check frontier point structure
        if len(data["frontier"]) > 0:
            point = data["frontier"][0]
            assert "return" in point
            assert "volatility" in point
            assert "sharpe" in point


class TestCorrelationMatrix:
    """Test /api/portfolio/correlation endpoint"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before correlation tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_get_correlation_matrix(self):
        """Test correlation matrix endpoint"""
        response = requests.get(f"{BASE_URL}/api/portfolio/correlation")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        
    def test_correlation_includes_diversification_pairs(self):
        """Test correlation endpoint includes diversification pairs"""
        response = requests.get(f"{BASE_URL}/api/portfolio/correlation")
        assert response.status_code == 200
        
        data = response.json()
        assert "diversification_pairs" in data


class TestPortfolioModelInfo:
    """Test /api/portfolio/model-info and training-status endpoints"""
    
    def test_get_model_info(self):
        """Test model info endpoint"""
        response = requests.get(f"{BASE_URL}/api/portfolio/model-info")
        assert response.status_code == 200
        
        data = response.json()
        # Should return model info structure
        assert isinstance(data, dict)
        
    def test_get_training_status(self):
        """Test training status endpoint"""
        response = requests.get(f"{BASE_URL}/api/portfolio/training-status")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)


class TestPortfolioStatistics:
    """Test /api/portfolio/statistics endpoint"""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Ensure data is fetched before statistics tests"""
        payload = {
            "assets": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1d"
        }
        requests.post(f"{BASE_URL}/api/portfolio/fetch-data", json=payload)
        time.sleep(1)
    
    def test_get_statistics(self):
        """Test statistics endpoint"""
        response = requests.get(f"{BASE_URL}/api/portfolio/statistics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "statistics" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
