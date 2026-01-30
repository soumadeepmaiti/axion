"""
Omni-Crypto Hybrid Deep Learning Trading System
FastAPI Backend Server
"""
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import asyncio
import numpy as np

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Import services
from services.data_pipeline import data_pipeline
from services.sentiment_service import sentiment_analyzer
from services.training_service import training_service
from ml_models.hybrid_model import hybrid_model

# Create the main app
app = FastAPI(
    title="Omni-Crypto Hybrid Trading System",
    description="Multi-Input Deep Learning for Crypto Trading",
    version="1.0.0"
)

# Create API router
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== Pydantic Models ==============

class PredictionRequest(BaseModel):
    symbol: str = "BTC/USDT"
    use_sentiment: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    direction: int
    direction_label: str
    probability: float
    confidence: float
    take_profit: float
    stop_loss: float
    risk_reward: float
    current_price: float
    sentiment_score: Optional[float] = None
    timestamp: str
    model_status: str
    model_accuracy: Optional[float] = None

class TrainingRequest(BaseModel):
    symbol: str = "BTC/USDT"
    epochs: int = 100
    batch_size: int = 32
    start_date: Optional[str] = None  # ISO format: 2024-01-01T00:00:00Z
    end_date: Optional[str] = None    # ISO format: 2025-01-01T00:00:00Z
    timeframe: str = "1h"

class SentimentRequest(BaseModel):
    text: str
    use_llm: bool = False

class MarketDataRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    limit: int = 100

class ModelConfigUpdate(BaseModel):
    lstm_units: Optional[int] = None
    gru_units: Optional[int] = None
    dense_units: Optional[int] = None
    dropout_rate: Optional[float] = None

# ============== Health & Status Endpoints ==============

@api_router.get("/")
async def root():
    return {"message": "Omni-Crypto Hybrid Trading System API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "data_pipeline": "active",
            "sentiment_analyzer": "active",
            "model": "active" if hybrid_model else "inactive",
            "database": "connected"
        }
    }

# ============== Market Data Endpoints ==============

@api_router.get("/market/symbols")
async def get_symbols():
    """Get available trading symbols"""
    return {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
    }

@api_router.post("/market/data")
async def get_market_data(request: MarketDataRequest):
    """Fetch OHLCV data for a symbol"""
    try:
        df = await data_pipeline.fetch_ohlcv(request.symbol, request.timeframe, request.limit)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        df = data_pipeline.calculate_technical_indicators(df)
        
        # Convert to JSON-safe format
        data = df.reset_index().to_dict('records')
        for row in data:
            row['timestamp'] = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            # Convert NaN to None
            for key, value in row.items():
                if isinstance(value, float) and np.isnan(value):
                    row[key] = None
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/market/latest/{symbol}")
async def get_latest_price(symbol: str):
    """Get latest price and indicators for a symbol"""
    try:
        symbol = symbol.replace("-", "/")
        data = await data_pipeline.get_latest_data(symbol)
        
        return {
            "symbol": symbol,
            "current_price": data['current_price'],
            "current_atr": data['current_atr'],
            "timestamp": data['timestamp']
        }
    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Sentiment Analysis Endpoints ==============

@api_router.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text"""
    try:
        result = await sentiment_analyzer.analyze(request.text, use_llm=request.use_llm)
        return {
            "text": request.text,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/sentiment/aggregate/{symbol}")
async def get_aggregate_sentiment(symbol: str):
    """Get aggregated sentiment for a symbol"""
    try:
        symbol = symbol.replace("-", "/")
        result = await sentiment_analyzer.get_aggregate_sentiment(symbol)
        
        # Store in database
        doc = {
            "id": str(uuid.uuid4()),
            **result,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.sentiment_history.insert_one(doc)
        
        return result
    except Exception as e:
        logger.error(f"Aggregate sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Model Endpoints ==============

@api_router.get("/model/summary")
async def get_model_summary():
    """Get model architecture summary"""
    return hybrid_model.get_model_summary()

@api_router.post("/model/build")
async def build_model(config: Optional[ModelConfigUpdate] = None):
    """Build or rebuild the model with optional config"""
    try:
        if hasattr(hybrid_model, 'build_model'):
            if config:
                if config.lstm_units:
                    hybrid_model.lstm_units = config.lstm_units
                if config.gru_units:
                    hybrid_model.gru_units = config.gru_units
                if config.dense_units:
                    hybrid_model.dense_units = config.dense_units
                if config.dropout_rate:
                    hybrid_model.dropout_rate = config.dropout_rate
                    
            hybrid_model.build_model()
            return {"status": "success", "message": "Model built successfully"}
        else:
            return {"status": "mock", "message": "Using mock model (TensorFlow not available)"}
    except Exception as e:
        logger.error(f"Model build error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Prediction Endpoints ==============

@api_router.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make a trading prediction using trained model"""
    try:
        # Get market data with mathematical analysis
        market_data = await data_pipeline.get_latest_data(request.symbol)
        
        # Get sentiment
        sentiment_data = None
        if request.use_sentiment:
            sentiment_data = await sentiment_analyzer.get_aggregate_sentiment(request.symbol)
        
        # Use trained model for prediction
        prediction = training_service.predict(market_data['features'])
        
        # Calculate TP/SL using mathematical analysis
        tp_sl = data_pipeline.calculate_tp_sl(
            market_data['current_price'],
            market_data['current_atr'],
            prediction['direction'],
            market_data.get('math_analysis')
        )
        
        # Build response
        response = PredictionResponse(
            symbol=request.symbol,
            direction=prediction['direction'],
            direction_label="LONG" if prediction['direction'] == 1 else "SHORT",
            probability=prediction['probability'],
            confidence=prediction['confidence'],
            take_profit=tp_sl['take_profit'],
            stop_loss=tp_sl['stop_loss'],
            risk_reward=tp_sl['risk_reward'],
            current_price=market_data['current_price'],
            sentiment_score=sentiment_data['aggregate_sentiment'] if sentiment_data else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_status=prediction['model_status'],
            model_accuracy=prediction.get('model_accuracy')
        )
        
        # Store prediction in database
        pred_doc = response.model_dump()
        pred_doc['id'] = str(uuid.uuid4())
        pred_doc['math_analysis'] = market_data.get('math_analysis')
        await db.predictions.insert_one(pred_doc)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/predictions/history")
async def get_prediction_history(symbol: Optional[str] = None, limit: int = 50):
    """Get prediction history"""
    try:
        query = {}
        if symbol:
            query['symbol'] = symbol.replace("-", "/")
            
        predictions = await db.predictions.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
        
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Training Endpoints ==============

@api_router.get("/training/status")
async def get_training_status():
    """Get current training status with learned patterns"""
    return training_service.get_status()

@api_router.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training with user-specified date range"""
    try:
        if training_service.training_status['is_training']:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Reset status
        training_service.reset_status()
        
        # Fetch training data for specified date range
        logger.info(f"Fetching data for {request.symbol} from {request.start_date} to {request.end_date}")
        data = await data_pipeline.get_training_data(
            request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe
        )
        
        if "error" in data:
            raise HTTPException(status_code=400, detail=data['error'])
        
        if len(data['features']) == 0:
            raise HTTPException(status_code=400, detail="No training data available for specified range")
        
        logger.info(f"Got {len(data['features'])} samples for training")
        
        # Start training in background
        async def train_task():
            result = await training_service.train_model(
                features=data['features'],
                labels=data['labels'],
                epochs=request.epochs,
                batch_size=request.batch_size,
                data_info=data['data_info']
            )
            # Save to database
            doc = {
                "id": str(uuid.uuid4()),
                "symbol": request.symbol,
                "result": result,
                "history": training_service.training_status['history'],
                "data_info": data['data_info'],
                "learned_patterns": training_service.get_learned_patterns(),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await db.training_history.insert_one(doc)
        
        background_tasks.add_task(train_task)
        
        return {
            "status": "started",
            "message": f"Training started for {request.symbol}",
            "config": {
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "timeframe": request.timeframe,
                "total_samples": len(data['features']),
                "date_range": data['data_info']['date_range']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/training/stop")
async def stop_training():
    """Stop current training"""
    return training_service.stop_training()

@api_router.get("/training/history")
async def get_training_history(limit: int = 10):
    """Get training history"""
    try:
        history = await db.training_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
        return {"count": len(history), "history": history}
    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Dashboard Stats ==============

@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Get counts
        prediction_count = await db.predictions.count_documents({})
        training_count = await db.training_history.count_documents({})
        
        # Get latest predictions for accuracy calculation
        recent_predictions = await db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(100).to_list(100)
        
        # Get model summary
        model_summary = hybrid_model.get_model_summary()
        
        return {
            "total_predictions": prediction_count,
            "training_sessions": training_count,
            "model_status": model_summary.get('status', 'unknown'),
            "is_trained": model_summary.get('is_trained', False),
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "recent_predictions": recent_predictions[:5],
            "training_status": training_service.get_status()
        }
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Omni-Crypto Hybrid Trading System")
    # Build model on startup
    if hasattr(hybrid_model, 'build_model'):
        try:
            hybrid_model.build_model()
            logger.info("Model built on startup")
        except Exception as e:
            logger.warning(f"Could not build model on startup: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
