"""
Sentiment Analysis Service
Uses FinBERT for financial sentiment, LLM for enhancement, and CryptoPanic for real news
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Fallback mock sentiment data
MOCK_SENTIMENTS = [
    {"text": "Bitcoin breaks through resistance, bulls are in control!", "source": "twitter", "sentiment": 0.85},
    {"text": "ETH network upgrades showing positive results", "source": "reddit", "sentiment": 0.72},
    {"text": "Crypto market shows signs of consolidation", "source": "news", "sentiment": 0.15},
    {"text": "Whale activity detected on major exchanges", "source": "twitter", "sentiment": -0.25},
    {"text": "Regulatory concerns weigh on crypto markets", "source": "news", "sentiment": -0.65},
]


class CryptoPanicClient:
    """Client for CryptoPanic News API"""
    
    BASE_URL = "https://cryptopanic.com/api/v1"
    
    def __init__(self):
        self.api_key = os.environ.get('CRYPTOPANIC_API_KEY')
        
    async def fetch_news(self, currencies: List[str] = None, kind: str = "news") -> List[Dict]:
        """Fetch news from CryptoPanic API"""
        if not self.api_key:
            logger.warning("CRYPTOPANIC_API_KEY not set")
            return []
            
        try:
            params = {
                "auth_token": self.api_key,
                "kind": kind,
                "public": "true"
            }
            
            if currencies:
                params["currencies"] = ",".join(currencies)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/posts/", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        logger.error(f"CryptoPanic API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"CryptoPanic fetch error: {e}")
            return []
    
    def extract_sentiment_from_votes(self, post: Dict) -> float:
        """Extract sentiment score from CryptoPanic votes"""
        votes = post.get("votes", {})
        positive = votes.get("positive", 0)
        negative = votes.get("negative", 0)
        liked = votes.get("liked", 0)
        disliked = votes.get("disliked", 0)
        
        total_positive = positive + liked
        total_negative = negative + disliked
        total = total_positive + total_negative
        
        if total == 0:
            # Use sentiment field if available
            sentiment_field = post.get("sentiment")
            if sentiment_field == "positive":
                return 0.5
            elif sentiment_field == "negative":
                return -0.5
            return 0.0
            
        return (total_positive - total_negative) / total


cryptopanic_client = CryptoPanicClient()


class SentimentAnalyzer:
    """Handles sentiment analysis using FinBERT and LLM enhancement"""
    
    def __init__(self):
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.llm_chat = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize FinBERT model and LLM"""
        if self._initialized:
            return
            
        try:
            # Initialize FinBERT
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_model.eval()
            logger.info("FinBERT loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}. Using fallback sentiment.")
            
        try:
            # Initialize LLM for enhancement
            from emergentintegrations.llm.chat import LlmChat, UserMessage
            
            api_key = os.environ.get('EMERGENT_LLM_KEY')
            if api_key:
                self.llm_chat = LlmChat(
                    api_key=api_key,
                    session_id="sentiment-analyzer",
                    system_message="You are a financial sentiment analyst. Analyze the given text and provide a sentiment score from -1 (very bearish) to 1 (very bullish), and a confidence score from 0 to 1. Respond ONLY with JSON: {\"sentiment\": number, \"confidence\": number, \"reasoning\": string}"
                ).with_model("openai", "gpt-5.2")
                logger.info("LLM initialized for sentiment enhancement")
            else:
                logger.warning("EMERGENT_LLM_KEY not found, LLM enhancement disabled")
                
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")
            
        self._initialized = True
    
    def analyze_with_finbert(self, text: str) -> Dict:
        """Analyze text sentiment using FinBERT"""
        if self.finbert_model is None or self.finbert_tokenizer is None:
            # Fallback to simple keyword-based sentiment
            return self._fallback_sentiment(text)
            
        try:
            import torch
            
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT classes: positive, negative, neutral
            positive = predictions[0][0].item()
            negative = predictions[0][1].item()
            neutral = predictions[0][2].item()
            
            # Convert to single score (-1 to 1)
            sentiment_score = positive - negative
            
            return {
                "sentiment": round(sentiment_score, 4),
                "positive": round(positive, 4),
                "negative": round(negative, 4),
                "neutral": round(neutral, 4),
                "confidence": round(max(positive, negative, neutral), 4)
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict:
        """Simple keyword-based fallback sentiment"""
        bullish_words = ['bull', 'buy', 'long', 'moon', 'pump', 'breakout', 'support', 'rally', 'surge', 'growth']
        bearish_words = ['bear', 'sell', 'short', 'dump', 'crash', 'resistance', 'decline', 'drop', 'fall', 'fear']
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return {"sentiment": 0, "confidence": 0.3, "positive": 0.33, "negative": 0.33, "neutral": 0.34}
            
        sentiment = (bullish_count - bearish_count) / total
        return {
            "sentiment": round(sentiment, 4),
            "confidence": round(min(total / 5, 1.0), 4),
            "positive": round(bullish_count / max(total, 1), 4),
            "negative": round(bearish_count / max(total, 1), 4),
            "neutral": round(1 - (bullish_count + bearish_count) / max(total * 2, 1), 4)
        }
    
    async def enhance_with_llm(self, text: str, finbert_result: Dict) -> Dict:
        """Enhance sentiment analysis with LLM"""
        if self.llm_chat is None:
            return finbert_result
            
        try:
            from emergentintegrations.llm.chat import UserMessage
            import json
            
            prompt = f"""Analyze this financial text:
"{text}"

FinBERT analysis: sentiment={finbert_result['sentiment']}, confidence={finbert_result['confidence']}

Provide your assessment as JSON with sentiment (-1 to 1) and confidence (0 to 1)."""
            
            message = UserMessage(text=prompt)
            response = await self.llm_chat.send_message(message)
            
            # Parse LLM response
            try:
                llm_result = json.loads(response)
                # Combine FinBERT and LLM results
                combined_sentiment = (finbert_result['sentiment'] * 0.6 + llm_result.get('sentiment', 0) * 0.4)
                combined_confidence = (finbert_result['confidence'] * 0.5 + llm_result.get('confidence', 0.5) * 0.5)
                
                return {
                    **finbert_result,
                    "sentiment": round(combined_sentiment, 4),
                    "confidence": round(combined_confidence, 4),
                    "llm_reasoning": llm_result.get('reasoning', ''),
                    "llm_enhanced": True
                }
            except json.JSONDecodeError:
                return {**finbert_result, "llm_raw": response, "llm_enhanced": False}
                
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
            return finbert_result
    
    async def analyze(self, text: str, use_llm: bool = True) -> Dict:
        """Full sentiment analysis pipeline"""
        await self.initialize()
        
        finbert_result = self.analyze_with_finbert(text)
        
        if use_llm and self.llm_chat is not None:
            return await self.enhance_with_llm(text, finbert_result)
            
        return finbert_result
    
    async def analyze_batch(self, texts: List[str], use_llm: bool = False) -> List[Dict]:
        """Analyze multiple texts"""
        await self.initialize()
        results = []
        
        for text in texts:
            result = await self.analyze(text, use_llm=use_llm)
            results.append(result)
            
        return results
    
    async def get_aggregate_sentiment(self, symbol: str) -> Dict:
        """Get aggregated sentiment for a symbol (using real CryptoPanic data)"""
        await self.initialize()
        
        # Map symbol to CryptoPanic currency codes
        currency_map = {
            "BTC/USDT": ["BTC"],
            "ETH/USDT": ["ETH"],
            "BTC": ["BTC"],
            "ETH": ["ETH"]
        }
        currencies = currency_map.get(symbol, ["BTC"])
        
        # Fetch real news from CryptoPanic
        news_posts = await cryptopanic_client.fetch_news(currencies=currencies)
        
        sentiments = []
        
        if news_posts:
            # Process real news
            for post in news_posts[:10]:  # Limit to 10 most recent
                title = post.get("title", "")
                
                # Get sentiment from votes or analyze with FinBERT
                vote_sentiment = cryptopanic_client.extract_sentiment_from_votes(post)
                finbert_result = await self.analyze(title, use_llm=False)
                
                # Combine vote sentiment and FinBERT analysis
                combined_sentiment = (vote_sentiment * 0.4 + finbert_result['sentiment'] * 0.6)
                
                sentiments.append({
                    "text": title,
                    "source": post.get("source", {}).get("title", "CryptoPanic"),
                    "url": post.get("url", ""),
                    "sentiment": round(combined_sentiment, 4),
                    "confidence": finbert_result['confidence'],
                    "vote_sentiment": vote_sentiment,
                    "published_at": post.get("published_at", ""),
                    "kind": post.get("kind", "news")
                })
            logger.info(f"Fetched {len(sentiments)} real news items from CryptoPanic")
        else:
            # Fallback to mock data if API fails
            logger.warning("Using mock sentiment data (CryptoPanic unavailable)")
            for mock in MOCK_SENTIMENTS:
                result = await self.analyze(mock['text'], use_llm=False)
                result['source'] = mock['source']
                result['text'] = mock['text']
                sentiments.append(result)
        
        if not sentiments:
            return {
                "aggregate_sentiment": 0,
                "aggregate_confidence": 0,
                "sample_count": 0,
                "sources": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "none"
            }
        
        avg_sentiment = sum(s['sentiment'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        return {
            "symbol": symbol,
            "aggregate_sentiment": round(avg_sentiment, 4),
            "aggregate_confidence": round(avg_confidence, 4),
            "sample_count": len(sentiments),
            "samples": sentiments[:5],  # Return first 5 samples
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_source": "cryptopanic" if news_posts else "mock"
        }


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()
