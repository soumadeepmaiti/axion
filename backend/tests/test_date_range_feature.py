"""
Test suite for arbitrary historical date range fetching feature.
Tests the data-preview endpoint and advanced training with date ranges.
"""
import pytest
import requests
import os
from datetime import datetime, timedelta

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestDataPreviewEndpoint:
    """Tests for /api/training/data-preview endpoint"""
    
    def test_data_preview_basic(self):
        """Test basic data preview without date range"""
        response = requests.post(
            f"{BASE_URL}/api/training/data-preview",
            json={
                "symbol": "BTC/USDT",
                "timeframe": "1h"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "symbol" in data
        assert "timeframe" in data
        assert "estimated_candles" in data
        assert "estimated_training_samples" in data
        assert "earliest_available" in data
        assert "exchange" in data
        
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert data["exchange"] == "binance"
        print(f"Basic preview: {data['estimated_candles']} estimated candles")
    
    def test_data_preview_with_date_range_2022_2024(self):
        """Test data preview with 2-year date range (2022-2024)"""
        response = requests.post(
            f"{BASE_URL}/api/training/data-preview",
            json={
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2022-01-01T00:00:00Z",
                "end_date": "2024-01-01T00:00:00Z"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Verify date range is parsed correctly
        assert "2022-01-01" in data["start_date"]
        assert "2024-01-01" in data["end_date"]
        
        # 2 years of hourly data should be ~17520 candles (730 days * 24 hours)
        assert data["estimated_candles"] > 17000
        assert data["estimated_candles"] < 18000
        
        # Training samples should be less (due to warmup period)
        assert data["estimated_training_samples"] < data["estimated_candles"]
        
        print(f"2022-2024 range: {data['estimated_candles']} candles, {data['estimated_training_samples']} training samples")
    
    def test_data_preview_with_large_date_range(self):
        """Test data preview with large date range (2017-2026) - daily timeframe"""
        response = requests.post(
            f"{BASE_URL}/api/training/data-preview",
            json={
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2017-08-17T00:00:00Z",
                "end_date": "2026-01-01T00:00:00Z"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # ~8.5 years of daily data should be ~3000+ candles
        assert data["estimated_candles"] > 3000
        
        # Earliest available should be 2017-08-17 for BTC/USDT
        assert data["earliest_available"] == "2017-08-17"
        
        print(f"Large range (2017-2026): {data['estimated_candles']} daily candles")
    
    def test_data_preview_size_warning_for_large_dataset(self):
        """Test that size warning appears for very large datasets"""
        response = requests.post(
            f"{BASE_URL}/api/training/data-preview",
            json={
                "symbol": "BTC/USDT",
                "timeframe": "5m",  # 5-minute candles for large dataset
                "start_date": "2020-01-01T00:00:00Z",
                "end_date": "2024-01-01T00:00:00Z"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # 4 years of 5-minute data = ~420,000 candles
        # Should trigger size warning
        if data["estimated_candles"] > 50000:
            assert data["size_warning"] is not None
            print(f"Size warning triggered: {data['size_warning']}")
        else:
            print(f"No size warning for {data['estimated_candles']} candles")
    
    def test_data_preview_eth_symbol(self):
        """Test data preview for ETH/USDT"""
        response = requests.post(
            f"{BASE_URL}/api/training/data-preview",
            json={
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2024-01-01T00:00:00Z"
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["symbol"] == "ETH/USDT"
        assert data["estimated_candles"] > 8000  # ~1 year of hourly data
        
        print(f"ETH/USDT preview: {data['estimated_candles']} candles")


class TestAdvancedTrainingWithDateRange:
    """Tests for advanced training with arbitrary date ranges"""
    
    def test_advanced_training_status(self):
        """Test that advanced training status endpoint works"""
        response = requests.get(f"{BASE_URL}/api/training/advanced/status")
        assert response.status_code == 200
        data = response.json()
        
        # Verify status structure
        assert "is_training" in data
        assert "progress" in data
        print(f"Training status: is_training={data['is_training']}, progress={data['progress']}")
    
    def test_advanced_training_start_with_date_range(self):
        """Test starting advanced training with a date range (short test)"""
        # Use a small date range for quick test
        response = requests.post(
            f"{BASE_URL}/api/training/advanced/start",
            json={
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-06-01T00:00:00Z",
                "epochs": 2,  # Very few epochs for quick test
                "batch_size": 32,
                "network_type": "lstm",
                "mode": "pure_ml",
                "save_model": False
            }
        )
        
        # Should either start successfully or indicate already training
        assert response.status_code in [200, 400]
        data = response.json()
        
        if response.status_code == 200:
            assert "status" in data
            print(f"Training started: {data.get('status')}")
            
            # Check if samples count is reasonable for the date range
            if "samples" in data:
                # ~5 months of daily data = ~150 samples
                assert data["samples"] > 100
                print(f"Training with {data['samples']} samples")
        else:
            # Training might already be in progress
            print(f"Training response: {data}")


class TestExchangeStatus:
    """Tests for exchange configuration"""
    
    def test_exchange_status(self):
        """Test exchange status endpoint"""
        response = requests.get(f"{BASE_URL}/api/exchange/status")
        assert response.status_code == 200
        data = response.json()
        
        assert "active_exchange" in data
        assert "exchanges" in data
        assert data["active_exchange"] == "binance"
        
        print(f"Active exchange: {data['active_exchange']}")
        print(f"Available exchanges: {data.get('available_exchanges', [])}")


class TestHealthAndBasicEndpoints:
    """Basic health and endpoint tests"""
    
    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "services" in data
        print(f"Health check passed: {data['services']}")
    
    def test_market_symbols(self):
        """Test market symbols endpoint"""
        response = requests.get(f"{BASE_URL}/api/market/symbols")
        assert response.status_code == 200
        data = response.json()
        
        assert "symbols" in data
        assert "timeframes" in data
        assert "BTC/USDT" in data["symbols"]
        assert "1h" in data["timeframes"]
        assert "1d" in data["timeframes"]
        
        print(f"Available symbols: {data['symbols']}")
        print(f"Available timeframes: {data['timeframes']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
