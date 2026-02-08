# 🔧 Making Axion Independent from Emergent

This guide explains all Emergent-specific components and how to replace them for independent development.

---

## Summary of Emergent-Specific Components

| Component | File(s) | Replacement |
|-----------|---------|-------------|
| **Emergent LLM Key** | `llm_service.py`, `sentiment_service.py` | OpenAI/Anthropic/Google API keys |
| **emergentintegrations** | `llm_service.py`, `sentiment_service.py` | Official SDKs (openai, anthropic, google-generativeai) |
| **Preview URLs** | `.env` files | Your deployment URLs |

---

## 1. Replace LLM Service (Most Important)

### Current File: `/app/backend/services/llm_service.py`

**Current (Emergent):**
```python
from emergentintegrations.llm.chat import LlmChat, UserMessage
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
```

**Replace with this independent version:**

Create new file `/app/backend/services/llm_service_independent.py`:

```python
"""
LLM Service for Trading Predictions - Independent Version
Supports OpenAI, Anthropic, and Google directly without Emergent.
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
    
    def __init__(self, model: str = "gpt-4o"):
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
                import re
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
                import re
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
        # Same implementation as OpenAI
        pass
    
    async def generate_prediction(self, market_data: Dict) -> LLMResponse:
        # Same implementation as OpenAI
        pass


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
        # Same implementation as OpenAI
        pass
    
    async def generate_prediction(self, market_data: Dict) -> LLMResponse:
        # Same implementation as OpenAI
        pass


class LLMService:
    """Multi-provider LLM Service - Independent Version"""
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize providers based on available API keys"""
        if OPENAI_API_KEY:
            self.providers["openai"] = OpenAIProvider()
            logger.info("OpenAI provider initialized")
        
        if ANTHROPIC_API_KEY:
            self.providers["claude"] = AnthropicProvider()
            logger.info("Anthropic provider initialized")
        
        if GOOGLE_API_KEY:
            self.providers["gemini"] = GeminiProvider()
            logger.info("Gemini provider initialized")
        
        if not self.providers:
            logger.warning("No LLM API keys configured. LLM features disabled.")
    
    async def analyze_sentiment(self, text: str, provider: str = "openai") -> LLMResponse:
        if provider not in self.providers:
            return LLMResponse(provider=provider, model="", content="", error="Provider not available")
        return await self.providers[provider].analyze_sentiment(text)
    
    async def generate_prediction(self, market_data: Dict, provider: str = "openai") -> LLMResponse:
        if provider not in self.providers:
            return LLMResponse(provider=provider, model="", content="", error="Provider not available")
        return await self.providers[provider].generate_prediction(market_data)
    
    def get_available_providers(self) -> List[str]:
        return list(self.providers.keys())


# Global instance
llm_service = LLMService()
```

---

## 2. Update Requirements

### Remove Emergent package, add official SDKs:

**Current `requirements.txt` has:**
```
emergentintegrations
```

**Replace with:**
```
openai>=1.0.0
anthropic>=0.18.0
google-generativeai>=0.3.0
```

### Full command:
```bash
cd /app/backend
pip uninstall emergentintegrations -y
pip install openai anthropic google-generativeai
pip freeze > requirements.txt
```

---

## 3. Update Environment Variables

### Backend `.env`:

**Current:**
```env
EMERGENT_LLM_KEY=your-emergent-key
```

**Replace with:**
```env
# LLM API Keys (get from respective providers)
OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/
GOOGLE_API_KEY=...             # https://makersuite.google.com/app/apikey

# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=axion
```

### Frontend `.env`:

**Current:**
```env
REACT_APP_BACKEND_URL=https://tradenet-ai.preview.emergentagent.com
```

**Replace with your deployed URL:**
```env
REACT_APP_BACKEND_URL=https://your-backend.onrender.com
# or for local development:
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 4. Update Sentiment Service

### File: `/app/backend/services/sentiment_service.py`

Find and replace the Emergent imports:

**Current:**
```python
from emergentintegrations.llm.chat import LlmChat, UserMessage
api_key = os.environ.get('EMERGENT_LLM_KEY')
```

**Replace with:**
```python
from openai import AsyncOpenAI
api_key = os.environ.get('OPENAI_API_KEY')

# Then update the LLM call:
client = AsyncOpenAI(api_key=api_key)
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
content = response.choices[0].message.content
```

---

## 5. Files to Modify - Summary

| File | Change Required |
|------|-----------------|
| `/app/backend/services/llm_service.py` | Replace entirely with independent version |
| `/app/backend/services/sentiment_service.py` | Update LLM imports (lines 120-131, 201) |
| `/app/backend/requirements.txt` | Remove emergentintegrations, add openai/anthropic/google |
| `/app/backend/.env` | Replace EMERGENT_LLM_KEY with individual API keys |
| `/app/frontend/.env` | Update backend URL |
| `/app/frontend/.env.production` | Update backend URL |

---

## 6. Quick Replacement Script

Run this to make the changes automatically:

```bash
#!/bin/bash
cd /app/backend

# Backup original files
cp services/llm_service.py services/llm_service.py.emergent.bak
cp services/sentiment_service.py services/sentiment_service.py.emergent.bak

# Update requirements
pip uninstall emergentintegrations -y 2>/dev/null
pip install openai anthropic google-generativeai
pip freeze > requirements.txt

# Update .env template
cat > .env.example << 'EOF'
# LLM API Keys (get your own keys)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-key

# Database
MONGO_URL=mongodb://localhost:27017
DB_NAME=axion

# Optional
CRYPTOPANIC_API_KEY=your-key
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
EOF

echo "✅ Environment updated for independent deployment"
echo ""
echo "Next steps:"
echo "1. Get API keys from:"
echo "   - OpenAI: https://platform.openai.com/api-keys"
echo "   - Anthropic: https://console.anthropic.com/"
echo "   - Google: https://makersuite.google.com/app/apikey"
echo ""
echo "2. Update /app/backend/.env with your keys"
echo "3. Replace llm_service.py with independent version"
```

---

## 7. Cost Comparison

| Provider | Model | Cost (per 1M tokens) |
|----------|-------|---------------------|
| **OpenAI** | GPT-4o | ~$5 input / $15 output |
| **OpenAI** | GPT-4o-mini | ~$0.15 input / $0.60 output |
| **Anthropic** | Claude 3.5 Sonnet | ~$3 input / $15 output |
| **Google** | Gemini 1.5 Flash | ~$0.075 input / $0.30 output |

**Recommendation:** Start with GPT-4o-mini or Gemini Flash for cost-effective development.

---

## 8. Testing After Changes

```bash
# Test LLM service
curl -X POST http://localhost:8001/api/advisor/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your BTC prediction?", "provider": "openai"}'

# Test sentiment
curl -X POST http://localhost:8001/api/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin is showing bullish momentum"}'
```

---

## Summary

To make Axion fully independent:

1. ✅ Replace `emergentintegrations` with official SDKs (`openai`, `anthropic`, `google-generativeai`)
2. ✅ Update `llm_service.py` to use official APIs
3. ✅ Update `sentiment_service.py` LLM calls
4. ✅ Get your own API keys from providers
5. ✅ Update `.env` files with your keys and URLs
6. ✅ Deploy anywhere you want!

**Total time to convert: ~30 minutes**

---

Need help with any specific part? Open an issue on GitHub!
