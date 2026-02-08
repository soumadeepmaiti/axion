"""
LLM Service for Trading Predictions - Independent Version
Supports OpenAI, Anthropic, and Google directly without Emergent.
"""
import os
import json
import asyncio
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# API Keys - use your own keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")


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


class OpenAIProvider:
    """OpenAI GPT Provider"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.provider = "openai"
        self.model = model
        self.name = "openai"
        self.api_key = OPENAI_API_KEY
    
    async def _call_api(self, prompt: str, system_prompt: str = "") -> str:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}
-1 = extremely bearish, 0 = neutral, 1 = extremely bullish"""
            
            content = await self._call_api(f"Analyze sentiment: {text}", system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider,
                        model=self.model,
                        content=content,
                        sentiment_score=data.get("sentiment_score", 0),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""),
                        latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content=content,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"OpenAI sentiment error: {e}")
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content="",
                error=str(e)
            )
    
    async def generate_prediction(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are an expert crypto trading analyst. Analyze market data and provide trading signals.
Respond with JSON only:
{"signal": "LONG" or "SHORT" or "NEUTRAL", "confidence": <float 0-1>, "reasoning": "<analysis>", "entry_price": <float>, "stop_loss": <float>, "take_profit": <float>}"""
            
            prompt = f"Analyze this market data and provide trading recommendation:\n{json.dumps(market_data, indent=2)}"
            content = await self._call_api(prompt, system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider,
                        model=self.model,
                        content=content,
                        trading_signal=data.get("signal", "NEUTRAL"),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""),
                        latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content=content,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"OpenAI prediction error: {e}")
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content="",
                error=str(e)
            )
    
    async def chat(self, message: str, context: str = "") -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = f"""You are an expert crypto trading advisor. Help users with market analysis, trading strategies, and portfolio management.
            
Context: {context}""" if context else "You are an expert crypto trading advisor. Help users with market analysis, trading strategies, and portfolio management."
            
            content = await self._call_api(message, system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content=content,
                latency_ms=latency
            )
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return LLMResponse(
                provider=self.provider,
                model=self.model,
                content="",
                error=str(e)
            )


class AnthropicProvider:
    """Anthropic Claude Provider"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.provider = "anthropic"
        self.model = model
        self.name = "claude"
        self.api_key = ANTHROPIC_API_KEY
    
    async def _call_api(self, prompt: str, system_prompt: str = "") -> str:
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(api_key=self.api_key)
        
        response = await client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are a crypto market sentiment analyzer. 
Analyze the given text and respond with JSON only:
{"sentiment_score": <float -1 to 1>, "confidence": <float 0 to 1>, "reasoning": "<brief explanation>"}"""
            
            content = await self._call_api(f"Analyze sentiment: {text}", system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider,
                        model=self.model,
                        content=content,
                        sentiment_score=data.get("sentiment_score", 0),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""),
                        latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))
    
    async def generate_prediction(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are an expert crypto trading analyst. Respond with JSON:
{"signal": "LONG"/"SHORT"/"NEUTRAL", "confidence": <0-1>, "reasoning": "<analysis>"}"""
            
            prompt = f"Analyze: {json.dumps(market_data, indent=2)}"
            content = await self._call_api(prompt, system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider, model=self.model, content=content,
                        trading_signal=data.get("signal"), confidence=data.get("confidence"),
                        reasoning=data.get("reasoning"), latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))
    
    async def chat(self, message: str, context: str = "") -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = "You are an expert crypto trading advisor."
            if context:
                system_prompt += f"\n\nContext: {context}"
            content = await self._call_api(message, system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))


class GeminiProvider:
    """Google Gemini Provider"""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.provider = "gemini"
        self.model = model
        self.name = "gemini"
        self.api_key = GOOGLE_API_KEY
    
    async def _call_api(self, prompt: str, system_prompt: str = "") -> str:
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        return response.text
    
    async def analyze_sentiment(self, text: str) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are a crypto sentiment analyzer. Respond with JSON only:
{"sentiment_score": <-1 to 1>, "confidence": <0-1>, "reasoning": "<explanation>"}"""
            
            content = await self._call_api(f"Analyze: {text}", system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider, model=self.model, content=content,
                        sentiment_score=data.get("sentiment_score", 0),
                        confidence=data.get("confidence", 0.5),
                        reasoning=data.get("reasoning", ""), latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))
    
    async def generate_prediction(self, market_data: Dict) -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = """You are an expert crypto analyst. Respond with JSON:
{"signal": "LONG"/"SHORT"/"NEUTRAL", "confidence": <0-1>, "reasoning": "<analysis>"}"""
            
            content = await self._call_api(f"Analyze: {json.dumps(market_data)}", system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            try:
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return LLMResponse(
                        provider=self.provider, model=self.model, content=content,
                        trading_signal=data.get("signal"), confidence=data.get("confidence"),
                        reasoning=data.get("reasoning"), latency_ms=latency
                    )
            except json.JSONDecodeError:
                pass
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))
    
    async def chat(self, message: str, context: str = "") -> LLMResponse:
        start_time = datetime.now()
        try:
            system_prompt = "You are an expert crypto trading advisor."
            if context:
                system_prompt += f"\n\nContext: {context}"
            content = await self._call_api(message, system_prompt)
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return LLMResponse(provider=self.provider, model=self.model, content=content, latency_ms=latency)
        except Exception as e:
            return LLMResponse(provider=self.provider, model=self.model, content="", error=str(e))


class LLMService:
    """Multi-provider LLM Service - Independent Version"""
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize providers based on available API keys"""
        if OPENAI_API_KEY:
            self.providers["openai"] = OpenAIProvider()
            logger.info("✓ OpenAI provider initialized")
        
        if ANTHROPIC_API_KEY:
            self.providers["claude"] = AnthropicProvider()
            logger.info("✓ Anthropic provider initialized")
        
        if GOOGLE_API_KEY:
            self.providers["gemini"] = GeminiProvider()
            logger.info("✓ Gemini provider initialized")
        
        if not self.providers:
            logger.warning("⚠ No LLM API keys configured. LLM features disabled.")
            logger.warning("  Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in .env")
    
    async def analyze_sentiment(self, text: str, provider: str = "openai") -> LLMResponse:
        if provider not in self.providers:
            available = list(self.providers.keys())
            if available:
                provider = available[0]
            else:
                return LLMResponse(provider=provider, model="", content="", error="No LLM providers available")
        return await self.providers[provider].analyze_sentiment(text)
    
    async def generate_prediction(self, market_data: Dict, provider: str = "openai") -> LLMResponse:
        if provider not in self.providers:
            available = list(self.providers.keys())
            if available:
                provider = available[0]
            else:
                return LLMResponse(provider=provider, model="", content="", error="No LLM providers available")
        return await self.providers[provider].generate_prediction(market_data)
    
    async def chat(self, message: str, provider: str = "openai", context: str = "") -> LLMResponse:
        if provider not in self.providers:
            available = list(self.providers.keys())
            if available:
                provider = available[0]
            else:
                return LLMResponse(provider=provider, model="", content="", error="No LLM providers available")
        return await self.providers[provider].chat(message, context)
    
    def get_available_providers(self) -> List[str]:
        return list(self.providers.keys())
    
    def is_available(self) -> bool:
        return len(self.providers) > 0


# Global instance
llm_service = LLMService()
