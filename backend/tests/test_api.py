"""
Backend API Tests for Omni-Crypto Hybrid Deep Learning Trading System
Tests: Health, Market Data, Predictions, Training, Dashboard
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealthEndpoint:
    """Health check endpoint tests"""
    
    def test_health_check(self):
        """Test /api/health returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["data_pipeline"] == "active"
        assert data["services"]["database"] == "connected"
        print(f"Health check passed: {data}")

    def test_root_endpoint(self):
        """Test /api/ root endpoint"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        print(f"Root endpoint passed: {data}")


class TestMarketDataEndpoints:
    """Market data endpoint tests"""
    
    def test_get_symbols(self):
        """Test /api/market/symbols returns available symbols"""
        response = requests.get(f"{BASE_URL}/api/market/symbols")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert "BTC/USDT" in data["symbols"]
        assert "timeframes" in data
        print(f"Symbols endpoint passed: {data}")

    def test_get_market_data_btc(self):
        """Test /api/market/data with BTC/USDT"""
        payload = {
            "symbol": "BTC/USDT",
            "timeframe": "5m",
            "limit": 50
        }
        response = requests.post(f"{BASE_URL}/api/market/data", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert "data" in data
        assert len(data["data"]) > 0
        # Verify OHLCV data structure
        first_candle = data["data"][0]
        assert "open" in first_candle
        assert "high" in first_candle
        assert "low" in first_candle
        assert "close" in first_candle
        assert "volume" in first_candle
        print(f"Market data BTC passed: {data['count']} candles")

    def test_get_latest_price_btc(self):
        """Test /api/market/latest/BTC-USDT"""
        response = requests.get(f"{BASE_URL}/api/market/latest/BTC-USDT")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert "current_price" in data
        assert data["current_price"] is not None
        assert data["current_price"] > 0
        print(f"Latest price BTC passed: ${data['current_price']}")


class TestPredictionEndpoints:
    """Prediction endpoint tests"""
    
    def test_make_prediction_btc(self):
        """Test /api/predict with BTC/USDT"""
        payload = {
            "symbol": "BTC/USDT",
            "use_sentiment": True
        }
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC/USDT"
        assert "direction" in data
        assert data["direction"] in [0, 1]
        assert "direction_label" in data
        assert data["direction_label"] in ["LONG", "SHORT"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "confidence" in data
        assert "take_profit" in data
        assert "stop_loss" in data
        assert "risk_reward" in data
        assert "current_price" in data
        assert "model_status" in data
        print(f"Prediction passed: {data['direction_label']} with {data['probability']*100:.1f}% probability")

    def test_prediction_history(self):
        """Test /api/predictions/history"""
        response = requests.get(f"{BASE_URL}/api/predictions/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "predictions" in data
        print(f"Prediction history passed: {data['count']} predictions")


class TestTrainingEndpoints:
    """Training endpoint tests"""
    
    def test_training_status(self):
        """Test /api/training/status"""
        response = requests.get(f"{BASE_URL}/api/training/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_training" in data
        assert "current_epoch" in data
        assert "total_epochs" in data
        print(f"Training status passed: is_training={data['is_training']}")

    def test_start_training_pure_ml(self):
        """Test /api/training/start with pure_ml mode"""
        payload = {
            "symbol": "BTC/USDT",
            "epochs": 10,
            "batch_size": 32,
            "timeframe": "1h",
            "mode": "pure_ml",
            "strategies": [],
            "num_lstm_layers": 2,
            "lstm_units": [64, 32],
            "num_dense_layers": 2,
            "dense_units": [32, 16],
            "dropout_rate": 0.3,
            "use_attention": True,
            "use_batch_norm": True,
            "learning_rate": 0.001,
            "sequence_length": 50
        }
        response = requests.post(f"{BASE_URL}/api/training/start", json=payload)
        # Could be 200 (started) or 400 (already training)
        assert response.status_code in [200, 400]
        data = response.json()
        if response.status_code == 200:
            assert data["status"] == "started"
            assert data["config"]["mode"] == "pure_ml"
            print(f"Training started: {data['message']}")
            # Wait for training to complete
            time.sleep(5)
        else:
            print(f"Training already in progress: {data}")

    def test_start_training_mathematical(self):
        """Test /api/training/start with mathematical mode and strategies"""
        # First check if training is in progress
        status_response = requests.get(f"{BASE_URL}/api/training/status")
        status = status_response.json()
        
        if status.get("is_training"):
            print("Training in progress, waiting...")
            time.sleep(10)
        
        payload = {
            "symbol": "BTC/USDT",
            "epochs": 10,
            "batch_size": 32,
            "timeframe": "1h",
            "mode": "mathematical",
            "strategies": ["mean_reversion", "momentum", "rsi"],
            "num_lstm_layers": 2,
            "lstm_units": [64, 32],
            "num_dense_layers": 2,
            "dense_units": [32, 16],
            "dropout_rate": 0.3,
            "use_attention": True,
            "use_batch_norm": True,
            "learning_rate": 0.001,
            "sequence_length": 50
        }
        response = requests.post(f"{BASE_URL}/api/training/start", json=payload)
        assert response.status_code in [200, 400]
        data = response.json()
        if response.status_code == 200:
            assert data["status"] == "started"
            assert data["config"]["mode"] == "mathematical"
            assert "mean_reversion" in data["config"]["strategies"]
            print(f"Mathematical training started: {data['message']}")
        else:
            print(f"Training response: {data}")

    def test_training_history(self):
        """Test /api/training/history"""
        response = requests.get(f"{BASE_URL}/api/training/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "history" in data
        print(f"Training history passed: {data['count']} sessions")


class TestDashboardEndpoints:
    """Dashboard endpoint tests"""
    
    def test_dashboard_stats(self):
        """Test /api/dashboard/stats"""
        response = requests.get(f"{BASE_URL}/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "training_sessions" in data
        assert "model_status" in data
        assert "symbols" in data
        assert "BTC/USDT" in data["symbols"]
        print(f"Dashboard stats passed: {data['total_predictions']} predictions, {data['training_sessions']} training sessions")


class TestSentimentEndpoints:
    """Sentiment analysis endpoint tests"""
    
    def test_aggregate_sentiment(self):
        """Test /api/sentiment/aggregate/BTC-USDT"""
        response = requests.get(f"{BASE_URL}/api/sentiment/aggregate/BTC-USDT")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "aggregate_sentiment" in data
        # Sentiment should be between -1 and 1
        assert -1 <= data["aggregate_sentiment"] <= 1
        print(f"Aggregate sentiment passed: {data['aggregate_sentiment']}")


class TestModelEndpoints:
    """Model endpoint tests"""
    
    def test_model_summary(self):
        """Test /api/model/summary"""
        response = requests.get(f"{BASE_URL}/api/model/summary")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"Model summary passed: {data}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
