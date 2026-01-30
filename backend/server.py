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

# Utility function to sanitize float values for JSON serialization
def sanitize_value(obj):
    """Recursively sanitize NaN and Inf values in nested structures"""
    if isinstance(obj, dict):
        return {k: sanitize_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_value(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_value(obj.tolist())
    return obj

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Import services
from services.data_pipeline import data_pipeline
from services.sentiment_service import sentiment_analyzer
from services.training_service import training_service
from ml_models.hybrid_model import hybrid_model

# Import advanced services
from services.advanced_data_pipeline import advanced_data_pipeline
from services.advanced_training_service import advanced_training_service
from services.backtesting_service import backtesting_service
from ml_models.advanced_models import list_saved_models, load_model

# Create the main app
app = FastAPI(
    title="Omni-Crypto Hybrid Trading System",
    description="Multi-Input Deep Learning for Crypto Trading",
    version="1.0.0"
)

# CORS middleware - MUST be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1h"
    # Mode: "pure_ml", "mathematical", "hybrid"
    mode: str = "pure_ml"
    # Mathematical strategies to use
    strategies: List[str] = []
    # Network type: "lstm", "gru", "transformer", "cnn_lstm", "ensemble"
    network_type: str = "lstm"
    # Network architecture
    num_lstm_layers: int = 2
    lstm_units: List[int] = [128, 64]
    num_dense_layers: int = 2
    dense_units: List[int] = [64, 32]
    dropout_rate: float = 0.3
    use_attention: bool = True
    use_batch_norm: bool = True
    learning_rate: float = 0.001
    sequence_length: int = 50
    # Advanced options
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    lr_schedule: str = "reduce_plateau"  # "cosine", "step", "reduce_plateau"
    use_walk_forward: bool = False
    cv_folds: int = 5
    use_optuna: bool = False
    optuna_trials: int = 20
    class_balance_method: str = "class_weight"  # "class_weight", "smote"
    multi_timeframe: bool = False
    save_model: bool = True

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

class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 10000.0
    position_size: float = 0.1  # 10% of capital per trade
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2%
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4%
    max_hold_time: int = 24  # hours (0 = no limit)
    min_confidence: float = 0.6  # minimum prediction confidence
    commission: float = 0.001  # 0.1% per trade

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
            # Convert NaN and Inf to None for JSON compatibility
            for key, value in list(row.items()):
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
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
        
        # Handle NaN/Inf values for JSON serialization
        def sanitize_float(value):
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                return None
            return value
        
        return {
            "symbol": symbol,
            "current_price": sanitize_float(data['current_price']),
            "current_atr": sanitize_float(data['current_atr']),
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
        
        if market_data.get('error') or market_data['current_price'] == 0:
            raise HTTPException(status_code=503, detail="Market data temporarily unavailable. Please try again.")
        
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
        
        # Helper to sanitize float values
        def sanitize_float(value, default=0.0):
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                return default
            return float(value)
        
        # Build response
        response = PredictionResponse(
            symbol=request.symbol,
            direction=prediction['direction'],
            direction_label="LONG" if prediction['direction'] == 1 else "SHORT",
            probability=sanitize_float(prediction['probability'], 0.5),
            confidence=sanitize_float(prediction['confidence'], 0.5),
            take_profit=sanitize_float(tp_sl['take_profit']),
            stop_loss=sanitize_float(tp_sl['stop_loss']),
            risk_reward=sanitize_float(tp_sl['risk_reward'], 1.67),
            current_price=sanitize_float(market_data['current_price']),
            sentiment_score=sanitize_float(sentiment_data['aggregate_sentiment']) if sentiment_data else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_status=prediction['model_status'],
            model_accuracy=sanitize_float(prediction.get('model_accuracy'))
        )
        
        # Store prediction in database
        pred_doc = response.model_dump()
        pred_doc['id'] = str(uuid.uuid4())
        pred_doc['math_analysis'] = market_data.get('math_analysis')
        await db.predictions.insert_one(pred_doc)
        
        return response
        
    except HTTPException:
        raise
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
    return sanitize_value(training_service.get_status())

@api_router.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training with user configuration"""
    try:
        if training_service.training_status['is_training']:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        training_service.reset_status()
        
        # Fetch training data
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
            raise HTTPException(status_code=400, detail="No data available for specified range")
        
        # Build config from request
        config = {
            "mode": request.mode,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "strategies": request.strategies,
            "num_lstm_layers": request.num_lstm_layers,
            "lstm_units": request.lstm_units,
            "num_dense_layers": request.num_dense_layers,
            "dense_units": request.dense_units,
            "dropout_rate": request.dropout_rate,
            "use_attention": request.use_attention,
            "use_batch_norm": request.use_batch_norm,
            "learning_rate": request.learning_rate,
            "sequence_length": request.sequence_length
        }
        
        logger.info(f"Training config: {config}")
        
        # Start training in background
        async def train_task():
            result = await training_service.train_model(
                features=data['features'],
                labels=data['labels'],
                prices=data.get('prices', np.array([])),
                config=config,
                data_info=data['data_info']
            )
            doc = {
                "id": str(uuid.uuid4()),
                "symbol": request.symbol,
                "result": result,
                "history": training_service.training_status['history'],
                "data_info": data['data_info'],
                "config": config,
                "learned_patterns": training_service._learned_patterns,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await db.training_history.insert_one(doc)
        
        background_tasks.add_task(train_task)
        
        return {
            "status": "started",
            "message": f"Training started in {request.mode} mode",
            "config": {
                "mode": request.mode,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "timeframe": request.timeframe,
                "strategies": request.strategies,
                "network": {
                    "lstm_layers": request.num_lstm_layers,
                    "dense_layers": request.num_dense_layers,
                    "attention": request.use_attention
                },
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
    # Try both services
    result1 = training_service.stop_training()
    result2 = advanced_training_service.stop_training()
    return result2 if advanced_training_service.is_training else result1

@api_router.get("/training/history")
async def get_training_history(limit: int = 10):
    """Get training history"""
    try:
        history = await db.training_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
        return {"count": len(history), "history": history}
    except Exception as e:
        logger.error(f"Error fetching training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Advanced Training Endpoints ==============

@api_router.post("/training/advanced/start")
async def start_advanced_training(request: TrainingRequest):
    """Start advanced model training with all features"""
    try:
        from datetime import datetime as dt
        
        # Parse dates
        start_date = None
        end_date = None
        if request.start_date:
            start_date = dt.fromisoformat(request.start_date.replace('Z', '+00:00'))
        if request.end_date:
            end_date = dt.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Build config
        config = {
            "mode": request.mode,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "timeframe": request.timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "network_type": request.network_type,
            "strategies": request.strategies,
            "num_lstm_layers": request.num_lstm_layers,
            "lstm_units": request.lstm_units,
            "num_dense_layers": request.num_dense_layers,
            "dense_units": request.dense_units,
            "dropout_rate": request.dropout_rate,
            "use_attention": request.use_attention,
            "use_batch_norm": request.use_batch_norm,
            "learning_rate": request.learning_rate,
            "sequence_length": request.sequence_length,
            "use_early_stopping": request.use_early_stopping,
            "early_stopping_patience": request.early_stopping_patience,
            "lr_schedule": request.lr_schedule,
            "use_walk_forward": request.use_walk_forward,
            "cv_folds": request.cv_folds,
            "use_optuna": request.use_optuna,
            "optuna_trials": request.optuna_trials,
            "class_balance_method": request.class_balance_method,
            "multi_timeframe": request.multi_timeframe,
            "save_model": request.save_model
        }
        
        result = await advanced_training_service.start_training(request.symbol, config)
        return sanitize_value(result)
        
    except Exception as e:
        logger.error(f"Advanced training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/training/advanced/status")
async def get_advanced_training_status():
    """Get advanced training status"""
    return sanitize_value(advanced_training_service.get_status())

# ============== Model Management Endpoints ==============

@api_router.get("/models/saved")
async def get_saved_models():
    """Get list of saved models"""
    try:
        models = list_saved_models()
        return {"count": len(models), "models": sanitize_value(models)}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/models/load")
async def load_saved_model(model_path: str):
    """Load a previously saved model"""
    try:
        result = await advanced_training_service.load_saved_model(model_path)
        return sanitize_value(result)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== Backtesting Endpoints ==============

@api_router.post("/backtest/start")
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a backtest with the loaded model"""
    try:
        if backtesting_service.is_running:
            raise HTTPException(status_code=400, detail="Backtest already in progress")
        
        # Check if model is loaded BEFORE starting background task
        if advanced_training_service.model is None and advanced_training_service.ensemble_model is None:
            raise HTTPException(status_code=400, detail="No trained model loaded. Please train or load a model first.")
        
        async def run_backtest_task():
            result = await backtesting_service.run_backtest(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital,
                position_size=request.position_size,
                use_stop_loss=request.use_stop_loss,
                stop_loss_pct=request.stop_loss_pct,
                use_take_profit=request.use_take_profit,
                take_profit_pct=request.take_profit_pct,
                max_hold_time=request.max_hold_time,
                min_confidence=request.min_confidence,
                commission=request.commission
            )
            
            # Save to database
            if "result" in result:
                doc = {
                    "id": str(uuid.uuid4()),
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "config": request.dict(),
                    "result": sanitize_value(result['result']),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.backtest_history.insert_one(doc)
        
        background_tasks.add_task(run_backtest_task)
        
        return {
            "status": "started",
            "message": f"Backtest started for {request.symbol}",
            "config": request.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/backtest/status")
async def get_backtest_status():
    """Get current backtest status"""
    return sanitize_value(backtesting_service.get_status())

@api_router.get("/backtest/result")
async def get_backtest_result():
    """Get backtest result"""
    result = backtesting_service.get_result()
    if result is None:
        return {"status": "no_result", "message": "No backtest result available"}
    return sanitize_value(result)

@api_router.post("/backtest/stop")
async def stop_backtest():
    """Stop running backtest"""
    return backtesting_service.stop_backtest()

@api_router.get("/backtest/history")
async def get_backtest_history(limit: int = 10):
    """Get backtest history"""
    try:
        history = await db.backtest_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
        return {"count": len(history), "history": sanitize_value(history)}
    except Exception as e:
        logger.error(f"Error fetching backtest history: {e}")
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
        
        # Get training status and sanitize
        training_status = sanitize_value(training_service.get_status())
        
        return {
            "total_predictions": prediction_count,
            "training_sessions": training_count,
            "model_status": model_summary.get('status', 'unknown'),
            "is_trained": model_summary.get('is_trained', False),
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "recent_predictions": sanitize_value(recent_predictions[:5]),
            "training_status": training_status
        }
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

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
