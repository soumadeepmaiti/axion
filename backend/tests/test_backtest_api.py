"""
Backend API Tests for Backtesting Framework
Tests: Backtest Status, Start, Result, History, Stop endpoints
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestBacktestStatusEndpoint:
    """Backtest status endpoint tests"""
    
    def test_backtest_status_structure(self):
        """Test GET /api/backtest/status returns correct status structure"""
        response = requests.get(f"{BASE_URL}/api/backtest/status")
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "is_running" in data
        assert "progress" in data
        assert "current_date" in data
        assert "total_trades" in data
        assert "current_pnl" in data
        assert "start_time" in data
        assert "error" in data
        
        # Verify types
        assert isinstance(data["is_running"], bool)
        assert isinstance(data["progress"], (int, float))
        assert isinstance(data["total_trades"], int)
        assert isinstance(data["current_pnl"], (int, float))
        
        print(f"Backtest status structure test passed: {data}")


class TestBacktestResultEndpoint:
    """Backtest result endpoint tests"""
    
    def test_backtest_result_no_result(self):
        """Test GET /api/backtest/result returns no_result when no backtest completed"""
        response = requests.get(f"{BASE_URL}/api/backtest/result")
        assert response.status_code == 200
        data = response.json()
        
        # When no backtest has been run, should return no_result status
        # OR if a backtest has been run, should return result data
        if "status" in data and data["status"] == "no_result":
            assert "message" in data
            assert data["message"] == "No backtest result available"
            print(f"Backtest result (no result): {data}")
        else:
            # If there's a result, verify structure
            assert "symbol" in data
            assert "total_return" in data
            assert "sharpe_ratio" in data
            print(f"Backtest result (has result): symbol={data.get('symbol')}, return={data.get('total_return_percent')}%")


class TestBacktestHistoryEndpoint:
    """Backtest history endpoint tests"""
    
    def test_backtest_history_returns_array(self):
        """Test GET /api/backtest/history returns history array"""
        response = requests.get(f"{BASE_URL}/api/backtest/history")
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert "count" in data
        assert "history" in data
        assert isinstance(data["history"], list)
        assert isinstance(data["count"], int)
        assert data["count"] == len(data["history"])
        
        print(f"Backtest history test passed: {data['count']} records")
        
        # If there are history records, verify structure
        if data["count"] > 0:
            record = data["history"][0]
            assert "symbol" in record or "id" in record
            print(f"First history record: {record.get('symbol', 'N/A')}")


class TestBacktestStartEndpoint:
    """Backtest start endpoint tests"""
    
    def test_backtest_start_without_model(self):
        """Test POST /api/backtest/start validates model is loaded"""
        # First check if a backtest is already running
        status_response = requests.get(f"{BASE_URL}/api/backtest/status")
        status = status_response.json()
        
        if status.get("is_running"):
            print("Backtest already running, skipping start test")
            pytest.skip("Backtest already in progress")
        
        # Try to start a backtest
        payload = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 10000.0,
            "position_size": 0.1,
            "use_stop_loss": True,
            "stop_loss_pct": 0.02,
            "use_take_profit": True,
            "take_profit_pct": 0.04,
            "max_hold_time": 24,
            "min_confidence": 0.6,
            "commission": 0.001
        }
        
        response = requests.post(f"{BASE_URL}/api/backtest/start", json=payload)
        
        # Should either start (200) or fail with model not loaded error (400)
        assert response.status_code in [200, 400]
        data = response.json()
        
        if response.status_code == 200:
            assert data["status"] == "started"
            assert "config" in data
            print(f"Backtest started: {data['message']}")
            # Wait a bit and check status
            time.sleep(2)
            status_response = requests.get(f"{BASE_URL}/api/backtest/status")
            print(f"Status after start: {status_response.json()}")
        else:
            # Should indicate model not loaded
            assert "error" in data or "detail" in data
            error_msg = data.get("error") or data.get("detail", "")
            print(f"Backtest start failed (expected if no model): {error_msg}")


class TestBacktestStopEndpoint:
    """Backtest stop endpoint tests"""
    
    def test_backtest_stop_when_not_running(self):
        """Test POST /api/backtest/stop when no backtest is running"""
        response = requests.post(f"{BASE_URL}/api/backtest/stop")
        assert response.status_code == 200
        data = response.json()
        
        # Should indicate not running or stopping
        assert "status" in data
        assert data["status"] in ["not_running", "stopping"]
        print(f"Backtest stop test passed: {data}")


class TestSavedModelsEndpoint:
    """Saved models endpoint tests"""
    
    def test_get_saved_models(self):
        """Test GET /api/models/saved returns list of saved models"""
        response = requests.get(f"{BASE_URL}/api/models/saved")
        assert response.status_code == 200
        data = response.json()
        
        assert "count" in data
        assert "models" in data
        assert isinstance(data["models"], list)
        
        print(f"Saved models test passed: {data['count']} models")
        
        if data["count"] > 0:
            model = data["models"][0]
            assert "symbol" in model
            assert "path" in model
            print(f"First model: {model.get('symbol')} - {model.get('network_type')}")


class TestBacktestIntegration:
    """Integration tests for backtest flow"""
    
    def test_full_backtest_flow(self):
        """Test the full backtest flow: check status -> start -> monitor -> result"""
        # 1. Check initial status
        status_response = requests.get(f"{BASE_URL}/api/backtest/status")
        assert status_response.status_code == 200
        initial_status = status_response.json()
        print(f"Initial status: is_running={initial_status['is_running']}")
        
        if initial_status.get("is_running"):
            print("Backtest already running, waiting for completion...")
            # Wait for completion
            for _ in range(30):
                time.sleep(2)
                status = requests.get(f"{BASE_URL}/api/backtest/status").json()
                if not status.get("is_running"):
                    break
        
        # 2. Check if there are saved models
        models_response = requests.get(f"{BASE_URL}/api/models/saved")
        models = models_response.json()
        print(f"Available models: {models['count']}")
        
        # 3. Check result
        result_response = requests.get(f"{BASE_URL}/api/backtest/result")
        assert result_response.status_code == 200
        result = result_response.json()
        print(f"Result status: {result.get('status', 'has_result')}")
        
        # 4. Check history
        history_response = requests.get(f"{BASE_URL}/api/backtest/history")
        assert history_response.status_code == 200
        history = history_response.json()
        print(f"History count: {history['count']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
