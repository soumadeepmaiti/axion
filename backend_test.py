#!/usr/bin/env python3
"""
Backend API Testing for Hybrid Deep Learning Trading System
Tests all endpoints for functionality and integration
"""
import requests
import json
import sys
import time
from datetime import datetime

class HybridTradingAPITester:
    def __init__(self, base_url="https://cryptotrader-ai-13.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
            self.failed_tests.append({"test": name, "details": details})

    def test_health_endpoint(self):
        """Test /api/health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Health Check", True)
                    return True
                else:
                    self.log_test("Health Check", False, f"Status not healthy: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False

    def test_market_symbols(self):
        """Test /api/market/symbols endpoint"""
        try:
            response = requests.get(f"{self.api_url}/market/symbols", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "symbols" in data and "BTC/USDT" in data["symbols"]:
                    self.log_test("Market Symbols", True)
                    return True
                else:
                    self.log_test("Market Symbols", False, f"Missing expected symbols: {data}")
                    return False
            else:
                self.log_test("Market Symbols", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Market Symbols", False, f"Exception: {str(e)}")
            return False

    def test_market_data(self):
        """Test /api/market/data endpoint"""
        try:
            payload = {
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "limit": 50
            }
            response = requests.post(f"{self.api_url}/market/data", json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    # Check if data has required fields
                    first_row = data["data"][0]
                    required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
                    if all(field in first_row for field in required_fields):
                        self.log_test("Market Data (OHLCV)", True)
                        return True
                    else:
                        self.log_test("Market Data (OHLCV)", False, f"Missing required fields in data")
                        return False
                else:
                    self.log_test("Market Data (OHLCV)", False, f"No data returned: {data}")
                    return False
            else:
                self.log_test("Market Data (OHLCV)", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Market Data (OHLCV)", False, f"Exception: {str(e)}")
            return False

    def test_latest_price(self):
        """Test /api/market/latest/{symbol} endpoint"""
        try:
            response = requests.get(f"{self.api_url}/market/latest/BTC-USDT", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "current_price" in data and data["current_price"] > 0:
                    self.log_test("Latest Price", True)
                    return True
                else:
                    self.log_test("Latest Price", False, f"Invalid price data: {data}")
                    return False
            else:
                self.log_test("Latest Price", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Latest Price", False, f"Exception: {str(e)}")
            return False

    def test_sentiment_analysis(self):
        """Test /api/sentiment/analyze endpoint"""
        try:
            payload = {
                "text": "Bitcoin is showing strong bullish momentum with great fundamentals",
                "use_llm": False
            }
            response = requests.post(f"{self.api_url}/sentiment/analyze", json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "sentiment" in data["result"]:
                    self.log_test("Sentiment Analysis", True)
                    return True
                else:
                    self.log_test("Sentiment Analysis", False, f"Missing sentiment result: {data}")
                    return False
            else:
                self.log_test("Sentiment Analysis", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Sentiment Analysis", False, f"Exception: {str(e)}")
            return False

    def test_aggregate_sentiment(self):
        """Test /api/sentiment/aggregate/{symbol} endpoint"""
        try:
            response = requests.get(f"{self.api_url}/sentiment/aggregate/BTC-USDT", timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "aggregate_sentiment" in data and "sample_count" in data:
                    self.log_test("Aggregate Sentiment", True)
                    return True
                else:
                    self.log_test("Aggregate Sentiment", False, f"Missing aggregate data: {data}")
                    return False
            else:
                self.log_test("Aggregate Sentiment", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Aggregate Sentiment", False, f"Exception: {str(e)}")
            return False

    def test_model_summary(self):
        """Test /api/model/summary endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model/summary", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "status" in data:
                    self.log_test("Model Summary", True)
                    return True
                else:
                    self.log_test("Model Summary", False, f"Missing status in response: {data}")
                    return False
            else:
                self.log_test("Model Summary", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Model Summary", False, f"Exception: {str(e)}")
            return False

    def test_prediction(self):
        """Test /api/predict endpoint"""
        try:
            payload = {
                "symbol": "BTC/USDT",
                "use_sentiment": True
            }
            response = requests.post(f"{self.api_url}/predict", json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["direction", "direction_label", "probability", "confidence", 
                                 "take_profit", "stop_loss", "current_price"]
                if all(field in data for field in required_fields):
                    # Validate direction is 0 or 1
                    if data["direction"] in [0, 1] and data["direction_label"] in ["LONG", "SHORT"]:
                        self.log_test("Prediction with TP/SL", True)
                        return True
                    else:
                        self.log_test("Prediction with TP/SL", False, f"Invalid direction values: {data}")
                        return False
                else:
                    self.log_test("Prediction with TP/SL", False, f"Missing required fields: {data}")
                    return False
            else:
                self.log_test("Prediction with TP/SL", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Prediction with TP/SL", False, f"Exception: {str(e)}")
            return False

    def test_prediction_history(self):
        """Test /api/predictions/history endpoint"""
        try:
            response = requests.get(f"{self.api_url}/predictions/history?limit=10", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "count" in data and "predictions" in data:
                    self.log_test("Prediction History", True)
                    return True
                else:
                    self.log_test("Prediction History", False, f"Missing expected fields: {data}")
                    return False
            else:
                self.log_test("Prediction History", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Prediction History", False, f"Exception: {str(e)}")
            return False

    def test_training_status(self):
        """Test /api/training/status endpoint"""
        try:
            response = requests.get(f"{self.api_url}/training/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "is_training" in data:
                    self.log_test("Training Status", True)
                    return True
                else:
                    self.log_test("Training Status", False, f"Missing training status: {data}")
                    return False
            else:
                self.log_test("Training Status", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Training Status", False, f"Exception: {str(e)}")
            return False

    def test_training_history(self):
        """Test /api/training/history endpoint"""
        try:
            response = requests.get(f"{self.api_url}/training/history?limit=5", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "count" in data and "history" in data:
                    self.log_test("Training History", True)
                    return True
                else:
                    self.log_test("Training History", False, f"Missing expected fields: {data}")
                    return False
            else:
                self.log_test("Training History", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Training History", False, f"Exception: {str(e)}")
            return False

    def test_dashboard_stats(self):
        """Test /api/dashboard/stats endpoint"""
        try:
            response = requests.get(f"{self.api_url}/dashboard/stats", timeout=15)
            if response.status_code == 200:
                data = response.json()
                expected_fields = ["total_predictions", "training_sessions", "model_status", "symbols"]
                if all(field in data for field in expected_fields):
                    self.log_test("Dashboard Stats", True)
                    return True
                else:
                    self.log_test("Dashboard Stats", False, f"Missing expected fields: {data}")
                    return False
            else:
                self.log_test("Dashboard Stats", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Dashboard Stats", False, f"Exception: {str(e)}")
            return False

    def test_start_training(self):
        """Test /api/training/start endpoint (quick test)"""
        try:
            payload = {
                "symbol": "BTC/USDT",
                "epochs": 2,  # Very small for testing
                "batch_size": 16,
                "lookback_days": 1
            }
            response = requests.post(f"{self.api_url}/training/start", json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if "status" in data and data["status"] == "started":
                    self.log_test("Start Training", True)
                    # Stop training immediately
                    time.sleep(1)
                    requests.post(f"{self.api_url}/training/stop", timeout=10)
                    return True
                else:
                    self.log_test("Start Training", False, f"Training not started: {data}")
                    return False
            elif response.status_code == 400:
                # Training might already be in progress
                data = response.json()
                if "already in progress" in data.get("detail", ""):
                    self.log_test("Start Training", True, "Training already in progress (expected)")
                    return True
                else:
                    self.log_test("Start Training", False, f"Bad request: {data}")
                    return False
            else:
                self.log_test("Start Training", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Start Training", False, f"Exception: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Hybrid Trading System API Tests")
        print("=" * 60)
        
        # Core health and market data tests
        self.test_health_endpoint()
        self.test_market_symbols()
        self.test_market_data()
        self.test_latest_price()
        
        # Sentiment analysis tests
        self.test_sentiment_analysis()
        self.test_aggregate_sentiment()
        
        # Model and prediction tests
        self.test_model_summary()
        self.test_prediction()
        self.test_prediction_history()
        
        # Training tests
        self.test_training_status()
        self.test_training_history()
        self.test_start_training()
        
        # Dashboard tests
        self.test_dashboard_stats()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.failed_tests:
            print("\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80  # Consider 80%+ as passing

def main():
    tester = HybridTradingAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())