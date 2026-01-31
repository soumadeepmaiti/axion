"""
LLM Service for Trading Predictions
Supports multiple LLM providers via Emergent LLM Key using emergentintegrations library.
"""
import os
import json
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Emergent LLM Key for supported providers
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")

@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None
    trading_signal: Optional[str] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class EmergentLLMProvider:
    """Base LLM Provider using Emergent Integration"""
    
    PROVIDER_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-4-sonnet-20250514",
        "gemini": "gemini-2.0-flash"
    }
    
    def __init__(self, provider: str, model: Optional[str] = None):
        self.provider = provider
        self.model = model or self.PROVIDER_MODELS.get(provider, "gpt-4o")
        self.name = provider
        self.api_key = EMERGENT_LLM_KEY
    
    async def _call_api(self, prompt: str, system_prompt: str = "") -> str:
        """Call the LLM API using emergentintegrations"""
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        
        session_id = f"trading-{self.provider}-{uuid.uuid4().hex[:8]}"
        
        chat = LlmChat(
            api_key=self.api_key,
            session_id=session_id,
            system_message=system_prompt or "You are a helpful assistant."
        ).with_model(self.provider, self.model)
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        return response
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""
            
            content = await self._call_api(f"Analyze sentiment: {text}", system_prompt)
            
            try:
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.name,
                        model=self.model,
                        content=content,
                        sentiment_score=data.get("sentiment_score", 0),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""),
                        latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
            except (json.JSONDecodeError, AttributeError):
                pass
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                sentiment_score=0,
                confidence=0.3,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"{self.name} sentiment error: {e}")
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def get_trading_signal(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""
            
            content = await self._call_api(f"Market data: {json.dumps(market_data)}", system_prompt)
            
            try:
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.name,
                        model=self.model,
                        content=content,
                        trading_signal=data.get("signal", "HOLD"),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""),
                        latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
            except (json.JSONDecodeError, AttributeError):
                pass
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                trading_signal="HOLD",
                confidence=0.3,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"{self.name} signal error: {e}")
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def chat(self, message: str, context: Optional[str] = None) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are an expert crypto trading advisor. Provide helpful, accurate analysis about cryptocurrency markets.
Include relevant technical analysis, market sentiment, and risk considerations in your responses.
Be concise but thorough. Always remind users that this is not financial advice."""
            
            if context:
                system_prompt += f"\n\nCurrent market context:\n{context}"
            
            content = await self._call_api(message, system_prompt)
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"{self.name} chat error: {e}")
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class LLMService:
    """
    Unified LLM Service for Trading Predictions
    Uses Emergent LLM Key for OpenAI, Claude, and Gemini
    """
    
    def __init__(self):
        self.providers: Dict[str, EmergentLLMProvider] = {}
        self._init_default_providers()
    
    def _init_default_providers(self):
        """Initialize default providers with Emergent LLM Key"""
        if not EMERGENT_LLM_KEY:
            logger.warning("EMERGENT_LLM_KEY not set. LLM features will be limited.")
            return
            
        try:
            self.providers["openai"] = EmergentLLMProvider("openai", "gpt-4o")
            self.providers["claude"] = EmergentLLMProvider("anthropic", "claude-4-sonnet-20250514")
            self.providers["gemini"] = EmergentLLMProvider("gemini", "gemini-2.0-flash")
            logger.info(f"LLM Service initialized with Emergent providers: {list(self.providers.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
    
    def configure(self, api_keys: Dict[str, str]):
        """Configure additional providers with custom API keys (future use)"""
        pass
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    async def analyze_sentiment(self, text: str, providers: Optional[List[str]] = None) -> Dict:
        """Analyze sentiment using specified providers"""
        if not self.providers:
            return {"error": "No LLM providers available", "sentiment_score": 0, "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        if not target_providers:
            return {"error": "No valid providers specified", "sentiment_score": 0, "results": []}
        
        tasks = [self.providers[p].analyze_sentiment(text) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, LLMResponse) and result.error is None:
                valid_results.append(result)
        
        if not valid_results:
            return {"error": "All providers failed", "sentiment_score": 0, "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]}
        
        total_weight = sum(r.confidence or 0.5 for r in valid_results)
        avg_sentiment = sum((r.sentiment_score or 0) * (r.confidence or 0.5) for r in valid_results) / total_weight if total_weight > 0 else 0
        avg_confidence = total_weight / len(valid_results)
        
        return {
            "sentiment_score": avg_sentiment,
            "confidence": avg_confidence,
            "providers_used": len(valid_results),
            "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]
        }
    
    async def get_ensemble_signal(self, market_data: Dict, providers: Optional[List[str]] = None) -> Dict:
        """Get trading signals from multiple LLMs and aggregate via voting"""
        if not self.providers:
            return {"error": "No LLM providers available", "signal": "HOLD", "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        if not target_providers:
            return {"error": "No valid providers specified", "signal": "HOLD", "results": []}
        
        tasks = [self.providers[p].get_trading_signal(market_data) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        confidences = {"BUY": [], "SELL": [], "HOLD": []}
        valid_results = []
        
        for result in results:
            if isinstance(result, LLMResponse) and result.error is None and result.trading_signal:
                signal = result.trading_signal.upper()
                if signal in votes:
                    votes[signal] += 1
                    confidences[signal].append(result.confidence or 0.5)
                    valid_results.append(result)
        
        if not valid_results:
            return {"error": "All providers failed", "signal": "HOLD", "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]}
        
        winning_signal = max(votes, key=votes.get)
        winning_count = votes[winning_signal]
        total_votes = sum(votes.values())
        
        consensus = winning_count / total_votes if total_votes > 0 else 0
        avg_confidence = sum(confidences[winning_signal]) / len(confidences[winning_signal]) if confidences[winning_signal] else 0.5
        
        return {
            "signal": winning_signal,
            "consensus": consensus,
            "confidence": avg_confidence,
            "votes": votes,
            "providers_used": len(valid_results),
            "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]
        }
    
    async def chat(self, message: str, provider: str = "openai", context: Optional[str] = None) -> Dict:
        """Chat with a specific LLM provider"""
        if provider not in self.providers:
            available = list(self.providers.keys())
            if not available:
                return {"error": "No LLM providers available", "content": ""}
            provider = available[0]
        
        result = await self.providers[provider].chat(message, context)
        return asdict(result)
    
    async def multi_chat(self, message: str, providers: Optional[List[str]] = None, context: Optional[str] = None) -> Dict:
        """Get responses from multiple LLMs for comparison"""
        if not self.providers:
            return {"error": "No LLM providers available", "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        tasks = [self.providers[p].chat(message, context) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]
        }


# Global instance - auto-initialized with Emergent LLM Key
llm_service = LLMService()
