"""
LLM Service for Trading Predictions
Supports multiple LLM providers for sentiment analysis, ensemble voting, and chat advisory.
Uses Emergent LLM Key for OpenAI, Claude, and Gemini integrations.
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Emergent LLM Key for supported providers
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "sk-emergent-0E3B0Bc93CaB82eD23")

@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    sentiment_score: Optional[float] = None  # -1 to 1 (bearish to bullish)
    confidence: Optional[float] = None
    trading_signal: Optional[str] = None  # 'BUY', 'SELL', 'HOLD'
    reasoning: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class EmergentOpenAIProvider:
    """OpenAI Provider using Emergent Integration"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.name = "openai"
        self.api_key = EMERGENT_LLM_KEY
    
    async def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> str:
        from emergentintegrations.llm.openai import generate_text
        
        # Convert messages to single prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
        
        response = generate_text(
            api_key=self.api_key,
            prompt=prompt.strip(),
            model=self.model
        )
        return response
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""},
                {"role": "user", "content": f"Analyze sentiment: {text}"}
            ]
            
            content = await self._call_api(messages)
            
            try:
                # Try to extract JSON from response
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
            logger.error(f"OpenAI sentiment error: {e}")
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
            messages = [
                {"role": "system", "content": """You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""},
                {"role": "user", "content": f"Market data: {json.dumps(market_data)}"}
            ]
            
            content = await self._call_api(messages)
            
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
            logger.error(f"OpenAI signal error: {e}")
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
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            content = await self._call_api(messages, temperature=0.7)
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class EmergentClaudeProvider:
    """Claude Provider using Emergent Integration"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.name = "claude"
        self.api_key = EMERGENT_LLM_KEY
    
    async def _call_api(self, prompt: str, system: str = "") -> str:
        from emergentintegrations.llm.anthropic import generate_text
        
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        response = generate_text(
            api_key=self.api_key,
            prompt=full_prompt,
            model=self.model
        )
        return response
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system = """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""
            
            content = await self._call_api(f"Analyze sentiment: {text}", system=system)
            
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
            logger.error(f"Claude sentiment error: {e}")
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
            system = """You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""
            
            content = await self._call_api(f"Market data: {json.dumps(market_data)}", system=system)
            
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
            logger.error(f"Claude signal error: {e}")
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
            system = """You are an expert crypto trading advisor. Provide helpful, accurate analysis about cryptocurrency markets.
Include relevant technical analysis, market sentiment, and risk considerations in your responses.
Be concise but thorough. Always remind users that this is not financial advice."""
            
            if context:
                system += f"\n\nCurrent market context:\n{context}"
            
            content = await self._call_api(message, system=system)
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"Claude chat error: {e}")
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class EmergentGeminiProvider:
    """Gemini Provider using Emergent Integration"""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.name = "gemini"
        self.api_key = EMERGENT_LLM_KEY
    
    async def _call_api(self, prompt: str) -> str:
        from emergentintegrations.llm.gemini import generate_text
        
        response = generate_text(
            api_key=self.api_key,
            prompt=prompt,
            model=self.model
        )
        return response
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            prompt = f"""You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish

Analyze sentiment: {text}"""
            
            content = await self._call_api(prompt)
            
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
            logger.error(f"Gemini sentiment error: {e}")
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
            prompt = f"""You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}}

Market data: {json.dumps(market_data)}"""
            
            content = await self._call_api(prompt)
            
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
            logger.error(f"Gemini signal error: {e}")
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
            prompt = """You are an expert crypto trading advisor. Provide helpful, accurate analysis about cryptocurrency markets.
Include relevant technical analysis, market sentiment, and risk considerations in your responses.
Be concise but thorough. Always remind users that this is not financial advice."""
            
            if context:
                prompt += f"\n\nCurrent market context:\n{context}"
            
            prompt += f"\n\nUser question: {message}"
            
            content = await self._call_api(prompt)
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
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
        self.providers: Dict[str, Any] = {}
        self._init_default_providers()
    
    def _init_default_providers(self):
        """Initialize default providers with Emergent LLM Key"""
        try:
            self.providers["openai"] = EmergentOpenAIProvider("gpt-4o")
            self.providers["claude"] = EmergentClaudeProvider("claude-sonnet-4-20250514")
            self.providers["gemini"] = EmergentGeminiProvider("gemini-2.0-flash")
            logger.info(f"LLM Service initialized with Emergent providers: {list(self.providers.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
    
    def configure(self, api_keys: Dict[str, str]):
        """Configure additional providers with custom API keys"""
        # Keep default Emergent providers, add custom ones if provided
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
        
        # Run all providers in parallel
        tasks = [self.providers[p].analyze_sentiment(text) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, LLMResponse) and result.error is None:
                valid_results.append(result)
        
        if not valid_results:
            return {"error": "All providers failed", "sentiment_score": 0, "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]}
        
        # Calculate weighted average sentiment
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
        
        # Run all providers in parallel
        tasks = [self.providers[p].get_trading_signal(market_data) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count votes
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
        
        # Determine winning signal
        winning_signal = max(votes, key=votes.get)
        winning_count = votes[winning_signal]
        total_votes = sum(votes.values())
        
        # Calculate consensus strength
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
        super().__init__(api_key, model)
        self.name = "openai"
        self.base_url = "https://api.openai.com/v1"
    
    async def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""},
                {"role": "user", "content": f"Analyze sentiment: {text}"}
            ]
            
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=data.get("sentiment_score", 0),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=0,
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            messages = [
                {"role": "system", "content": """You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""},
                {"role": "user", "content": f"Market data: {json.dumps(market_data)}"}
            ]
            
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal=data.get("signal", "HOLD"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal="HOLD",
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            result = await self._call_api(messages, temperature=0.7)
            content = result["choices"][0]["message"]["content"]
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class ClaudeProvider(LLMProvider):
    """Anthropic Claude Provider"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        self.name = "claude"
        self.base_url = "https://api.anthropic.com/v1"
    
    async def _call_api(self, messages: List[Dict], system: str = "", temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": self.model,
                    "max_tokens": 500,
                    "system": system,
                    "messages": messages,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system = """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""
            
            messages = [{"role": "user", "content": f"Analyze sentiment: {text}"}]
            
            result = await self._call_api(messages, system=system)
            content = result["content"][0]["text"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=data.get("sentiment_score", 0),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=0,
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            system = """You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""
            
            messages = [{"role": "user", "content": f"Market data: {json.dumps(market_data)}"}]
            
            result = await self._call_api(messages, system=system)
            content = result["content"][0]["text"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal=data.get("signal", "HOLD"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal="HOLD",
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            system = """You are an expert crypto trading advisor. Provide helpful, accurate analysis about cryptocurrency markets.
Include relevant technical analysis, market sentiment, and risk considerations in your responses.
Be concise but thorough. Always remind users that this is not financial advice."""
            
            if context:
                system += f"\n\nCurrent market context:\n{context}"
            
            messages = [{"role": "user", "content": message}]
            
            result = await self._call_api(messages, system=system, temperature=0.7)
            content = result["content"][0]["text"]
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class GeminiProvider(LLMProvider):
    """Google Gemini Provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__(api_key, model)
        self.name = "gemini"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def _call_api(self, prompt: str, temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": 500
                    }
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            prompt = f"""You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish

Analyze sentiment: {text}"""
            
            result = await self._call_api(prompt)
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            try:
                # Extract JSON from response
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
            prompt = f"""You are an expert crypto trader. Analyze the market data and provide a trading signal.
Respond with JSON only:
{{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}}

Market data: {json.dumps(market_data)}"""
            
            result = await self._call_api(prompt)
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
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
            prompt = """You are an expert crypto trading advisor. Provide helpful, accurate analysis about cryptocurrency markets.
Include relevant technical analysis, market sentiment, and risk considerations in your responses.
Be concise but thorough. Always remind users that this is not financial advice."""
            
            if context:
                prompt += f"\n\nCurrent market context:\n{context}"
            
            prompt += f"\n\nUser question: {message}"
            
            result = await self._call_api(prompt, temperature=0.7)
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class DeepSeekProvider(LLMProvider):
    """DeepSeek Provider"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(api_key, model)
        self.name = "deepseek"
        self.base_url = "https://api.deepseek.com/v1"
    
    async def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""},
                {"role": "user", "content": f"Analyze sentiment: {text}"}
            ]
            
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=data.get("sentiment_score", 0),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    sentiment_score=0,
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            messages = [
                {"role": "system", "content": """You are an expert crypto trader. Respond with JSON only:
{"signal": "BUY" or "SELL" or "HOLD", "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""},
                {"role": "user", "content": f"Market data: {json.dumps(market_data)}"}
            ]
            
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal=data.get("signal", "HOLD"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    content=content,
                    trading_signal="HOLD",
                    confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
        except Exception as e:
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
            system = """You are an expert crypto trading advisor. Be concise but thorough. This is not financial advice."""
            if context:
                system += f"\n\nMarket context:\n{context}"
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
            
            result = await self._call_api(messages, temperature=0.7)
            content = result["choices"][0]["message"]["content"]
            
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content=content,
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
                
        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                content="",
                error=str(e),
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )


class GrokProvider(LLMProvider):
    """xAI Grok Provider"""
    
    def __init__(self, api_key: str, model: str = "grok-beta"):
        super().__init__(api_key, model)
        self.name = "grok"
        self.base_url = "https://api.x.ai/v1"
    
    async def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """Crypto sentiment analyzer. JSON only:
{"sentiment_score": <-1 to 1>, "confidence": <0 to 1>, "reasoning": "<explanation>"}"""},
                {"role": "user", "content": f"Analyze: {text}"}
            ]
            
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name, model=self.model, content=content,
                    sentiment_score=data.get("sentiment_score", 0),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(
                    provider=self.name, model=self.model, content=content,
                    sentiment_score=0, confidence=0.3,
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
    
    async def get_trading_signal(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """Crypto trader. JSON only:
{"signal": "BUY"/"SELL"/"HOLD", "confidence": <0-1>, "reasoning": "<why>"}"""},
                {"role": "user", "content": f"Data: {json.dumps(market_data)}"}
            ]
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name, model=self.model, content=content,
                    trading_signal=data.get("signal", "HOLD"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(provider=self.name, model=self.model, content=content,
                                 trading_signal="HOLD", confidence=0.3,
                                 latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
    
    async def chat(self, message: str, context: Optional[str] = None) -> LLMResponse:
        start_time = datetime.now()
        try:
            system = "Expert crypto advisor. Be direct and insightful. Not financial advice."
            if context:
                system += f"\nContext: {context}"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
            result = await self._call_api(messages, temperature=0.7)
            return LLMResponse(
                provider=self.name, model=self.model,
                content=result["choices"][0]["message"]["content"],
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)


class KimiProvider(LLMProvider):
    """Moonshot Kimi Provider"""
    
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k"):
        super().__init__(api_key, model)
        self.name = "kimi"
        self.base_url = "https://api.moonshot.cn/v1"
    
    async def _call_api(self, messages: List[Dict], temperature: float = 0.3) -> Dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """Crypto sentiment analyzer. JSON:
{"sentiment_score": <-1 to 1>, "confidence": <0 to 1>, "reasoning": "<why>"}"""},
                {"role": "user", "content": f"Analyze: {text}"}
            ]
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name, model=self.model, content=content,
                    sentiment_score=data.get("sentiment_score", 0),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(provider=self.name, model=self.model, content=content,
                                 sentiment_score=0, confidence=0.3,
                                 latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
    
    async def get_trading_signal(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            messages = [
                {"role": "system", "content": """Trader. JSON: {"signal": "BUY"/"SELL"/"HOLD", "confidence": <0-1>, "reasoning": "<why>"}"""},
                {"role": "user", "content": f"Data: {json.dumps(market_data)}"}
            ]
            result = await self._call_api(messages)
            content = result["choices"][0]["message"]["content"]
            try:
                data = json.loads(content)
                return LLMResponse(
                    provider=self.name, model=self.model, content=content,
                    trading_signal=data.get("signal", "HOLD"),
                    confidence=data.get("confidence", 0.5),
                    reasoning=data.get("reasoning", ""),
                    latency_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            except json.JSONDecodeError:
                return LLMResponse(provider=self.name, model=self.model, content=content,
                                 trading_signal="HOLD", confidence=0.3,
                                 latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)
    
    async def chat(self, message: str, context: Optional[str] = None) -> LLMResponse:
        start_time = datetime.now()
        try:
            system = "Expert crypto advisor. Concise. Not financial advice."
            if context:
                system += f"\nContext: {context}"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
            result = await self._call_api(messages, temperature=0.7)
            return LLMResponse(
                provider=self.name, model=self.model,
                content=result["choices"][0]["message"]["content"],
                latency_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return LLMResponse(provider=self.name, model=self.model, content="", error=str(e),
                             latency_ms=(datetime.now() - start_time).total_seconds() * 1000)


class LLMService:
    """
    Unified LLM Service for Trading Predictions
    Manages multiple providers and provides ensemble capabilities
    """
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.api_keys: Dict[str, str] = {}
    
    def configure(self, api_keys: Dict[str, str]):
        """Configure providers with API keys"""
        self.api_keys = api_keys
        self.providers = {}
        
        # Initialize available providers
        if api_keys.get("openai"):
            self.providers["openai"] = OpenAIProvider(api_keys["openai"], api_keys.get("openai_model", "gpt-4o"))
        
        if api_keys.get("claude"):
            self.providers["claude"] = ClaudeProvider(api_keys["claude"], api_keys.get("claude_model", "claude-sonnet-4-20250514"))
        
        if api_keys.get("gemini"):
            self.providers["gemini"] = GeminiProvider(api_keys["gemini"], api_keys.get("gemini_model", "gemini-1.5-flash"))
        
        if api_keys.get("deepseek"):
            self.providers["deepseek"] = DeepSeekProvider(api_keys["deepseek"], api_keys.get("deepseek_model", "deepseek-chat"))
        
        if api_keys.get("grok"):
            self.providers["grok"] = GrokProvider(api_keys["grok"], api_keys.get("grok_model", "grok-beta"))
        
        if api_keys.get("kimi"):
            self.providers["kimi"] = KimiProvider(api_keys["kimi"], api_keys.get("kimi_model", "moonshot-v1-8k"))
        
        logger.info(f"LLM Service configured with providers: {list(self.providers.keys())}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured providers"""
        return list(self.providers.keys())
    
    async def analyze_sentiment(self, text: str, providers: Optional[List[str]] = None) -> Dict:
        """
        Analyze sentiment using specified providers (or all available)
        Returns aggregated sentiment with individual results
        """
        if not self.providers:
            return {"error": "No LLM providers configured", "sentiment_score": 0, "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        if not target_providers:
            return {"error": "No valid providers specified", "sentiment_score": 0, "results": []}
        
        # Run all providers in parallel
        tasks = [self.providers[p].analyze_sentiment(text) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, LLMResponse) and result.error is None:
                valid_results.append(result)
        
        if not valid_results:
            return {"error": "All providers failed", "sentiment_score": 0, "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]}
        
        # Calculate weighted average sentiment
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
        """
        Get trading signals from multiple LLMs and aggregate via voting
        """
        if not self.providers:
            return {"error": "No LLM providers configured", "signal": "HOLD", "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        if not target_providers:
            return {"error": "No valid providers specified", "signal": "HOLD", "results": []}
        
        # Run all providers in parallel
        tasks = [self.providers[p].get_trading_signal(market_data) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count votes
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
        
        # Determine winning signal
        winning_signal = max(votes, key=votes.get)
        winning_count = votes[winning_signal]
        total_votes = sum(votes.values())
        
        # Calculate consensus strength
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
        """
        Chat with a specific LLM provider
        """
        if provider not in self.providers:
            available = list(self.providers.keys())
            if not available:
                return {"error": "No LLM providers configured", "content": ""}
            provider = available[0]  # Use first available
        
        result = await self.providers[provider].chat(message, context)
        return asdict(result)
    
    async def multi_chat(self, message: str, providers: Optional[List[str]] = None, context: Optional[str] = None) -> Dict:
        """
        Get responses from multiple LLMs for comparison
        """
        if not self.providers:
            return {"error": "No LLM providers configured", "results": []}
        
        target_providers = providers or list(self.providers.keys())
        target_providers = [p for p in target_providers if p in self.providers]
        
        tasks = [self.providers[p].chat(message, context) for p in target_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": [asdict(r) for r in results if isinstance(r, LLMResponse)]
        }


# Global instance
llm_service = LLMService()
