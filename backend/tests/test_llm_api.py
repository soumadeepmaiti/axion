"""
LLM API Tests for Omni-Crypto Trading System
Tests all LLM endpoints: providers, chat, multi-chat, signal
"""
import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestLLMProviders:
    """Test LLM providers endpoint"""
    
    def test_get_providers_returns_three(self):
        """GET /api/llm/providers should return 3 providers"""
        response = requests.get(f"{BASE_URL}/api/llm/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) == 3
        assert "openai" in data["providers"]
        assert "claude" in data["providers"]
        assert "gemini" in data["providers"]
        
    def test_get_providers_returns_all_supported(self):
        """GET /api/llm/providers should return all_supported list"""
        response = requests.get(f"{BASE_URL}/api/llm/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert "all_supported" in data
        assert len(data["all_supported"]) >= 3


class TestLLMChat:
    """Test LLM chat endpoint with all providers"""
    
    def test_chat_openai(self):
        """POST /api/llm/chat with OpenAI should return valid response"""
        response = requests.post(
            f"{BASE_URL}/api/llm/chat",
            json={"message": "What is Bitcoin in one sentence?", "provider": "openai"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"
        assert data["content"] is not None
        assert len(data["content"]) > 10
        assert data["error"] is None
        assert data["latency_ms"] is not None
        assert data["latency_ms"] > 0
        
    def test_chat_claude(self):
        """POST /api/llm/chat with Claude should return valid response"""
        response = requests.post(
            f"{BASE_URL}/api/llm/chat",
            json={"message": "What is Ethereum in one sentence?", "provider": "claude"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["provider"] == "anthropic"
        assert "claude" in data["model"]
        assert data["content"] is not None
        assert len(data["content"]) > 10
        assert data["error"] is None
        
    def test_chat_gemini(self):
        """POST /api/llm/chat with Gemini should return valid response"""
        response = requests.post(
            f"{BASE_URL}/api/llm/chat",
            json={"message": "What is DeFi in one sentence?", "provider": "gemini"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["provider"] == "gemini"
        assert "gemini" in data["model"]
        assert data["content"] is not None
        assert len(data["content"]) > 10
        assert data["error"] is None


class TestLLMMultiChat:
    """Test LLM multi-chat endpoint"""
    
    def test_multi_chat_returns_all_providers(self):
        """POST /api/llm/multi-chat should return responses from all providers"""
        response = requests.post(
            f"{BASE_URL}/api/llm/multi-chat",
            json={"message": "What is crypto?"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3
        
        providers = [r["provider"] for r in data["results"]]
        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers
        
        # All should have content
        for result in data["results"]:
            assert result["content"] is not None
            assert len(result["content"]) > 10


class TestLLMEnsembleSignal:
    """Test LLM ensemble signal endpoint"""
    
    def test_ensemble_signal_returns_votes(self):
        """POST /api/llm/signal should return signal with votes from all providers"""
        response = requests.post(
            f"{BASE_URL}/api/llm/signal",
            json={"symbol": "BTC/USDT"}
        )
        assert response.status_code == 200
        
        data = response.json()
        
        # Check signal
        assert "signal" in data
        assert data["signal"] in ["BUY", "SELL", "HOLD"]
        
        # Check votes
        assert "votes" in data
        assert "BUY" in data["votes"]
        assert "SELL" in data["votes"]
        assert "HOLD" in data["votes"]
        
        # Check consensus
        assert "consensus" in data
        assert 0 <= data["consensus"] <= 1
        
        # Check providers used
        assert "providers_used" in data
        assert data["providers_used"] == 3
        
        # Check individual results
        assert "results" in data
        assert len(data["results"]) == 3
        
        for result in data["results"]:
            assert result["trading_signal"] in ["BUY", "SELL", "HOLD"]
            assert result["confidence"] is not None
            assert result["reasoning"] is not None


class TestLLMSentiment:
    """Test LLM sentiment analysis endpoint"""
    
    def test_sentiment_analysis(self):
        """POST /api/llm/sentiment should return sentiment analysis"""
        response = requests.post(
            f"{BASE_URL}/api/llm/sentiment",
            json={"text": "Bitcoin is going to the moon! Very bullish!"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment_score" in data
        assert "results" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
