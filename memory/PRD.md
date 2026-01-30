# Omni-Crypto Hybrid Deep Learning Trading System

## Original Problem Statement
Build a Multi-Input Hybrid Deep Learning System for crypto trading using Late Fusion strategy combining:
1. **Micro Data (1m, 5m, 10m OHLCV)**: Processed by Bi-LSTM for immediate momentum
2. **Macro Data (1h, 1d, 1y, 10y)**: Processed by Attention-based GRU for broader market context
3. **Sentiment Data (Twitter, Reddit, News)**: Processed by FinBERT for polarity/subjectivity

User wants two training modes:
- **Pure ML Mode**: Model discovers all patterns from data
- **Mathematical Modeling Mode**: Apply pre-defined quant strategies (Renaissance-style)

## User Choices
- Data Source: CCXT Binance US
- Trading Pairs: BTC/USDT, ETH/USDT
- Model: Fully trainable with user-configurable network depth
- Sentiment: FinBERT + CryptoPanic (MOCKED - invalid API key)
- Training Date Range: User-selectable

## Architecture
```
/app/
├── backend/
│   ├── server.py             # FastAPI endpoints
│   ├── ml_models/hybrid_model.py
│   └── services/
│       ├── data_pipeline.py    # CCXT data fetching
│       ├── sentiment_service.py # FinBERT + CryptoPanic (mocked)
│       └── training_service.py  # Dual-mode training (Pure ML + Math)
└── frontend/
    └── src/pages/
        ├── Dashboard.jsx   # Price charts, predictions
        ├── Training.jsx    # Network config, training modes
        └── Predictions.jsx # Prediction history
```

## What's Been Implemented (Jan 30, 2026)

### Backend - Advanced Training System
- ✅ FastAPI server with CORS middleware
- ✅ CCXT Binance US integration for live market data
- ✅ **Advanced Data Pipeline** (`advanced_data_pipeline.py`):
  - Multi-timeframe data support (5m, 15m, 1h, 4h, 1d)
  - Extended historical data (up to 2 years)
  - 50+ technical indicators using `ta` library
  - Market Regime Detection (Bull/Bear/Sideways)
  - Cross-Asset Correlation (BTC dominance, ETH/BTC ratio)
  - GARCH Volatility Features
  - Mock Order Book Features (bid-ask spread, order flow - ready for real API)
  - Mock On-Chain Metrics (active addresses, exchange flows - ready for real API)
- ✅ **Advanced Model Architectures** (`advanced_models.py`):
  - LSTM (Bi-directional with attention)
  - GRU (Bi-directional with attention)
  - Transformer (Multi-head attention)
  - CNN + LSTM Hybrid
  - Ensemble (LSTM + XGBoost + RandomForest)
  - Late Fusion support for multi-timeframe
- ✅ **Advanced Training Service** (`advanced_training_service.py`):
  - Walk-Forward Validation (time-series CV)
  - Class Balancing (SMOTE or class weights)
  - Optuna Hyperparameter Search
  - Learning Rate Scheduling (cosine, step, reduce_plateau)
  - Optional Early Stopping
  - **Model Persistence** (save/load trained models)
- ✅ **API Endpoints**:
  - `/api/training/advanced/start` - Start advanced training
  - `/api/training/advanced/status` - Get training status
  - `/api/models/saved` - List saved models
  - `/api/models/load` - Load saved model

### Frontend
- ✅ Dashboard with real-time price charts
- ✅ **Training Page** with:
  - Network Type selector (LSTM, GRU, Transformer, CNN+LSTM, Ensemble)
  - Network Layers configuration
  - Hyperparameters configuration
  - Start/Stop training buttons
  - Real-time training progress
  - Training history
- ✅ Predictions history page

## Core Requirements - Status
| Requirement | Status |
|-------------|--------|
| Multi-input hybrid model | ✅ Complete |
| Real-time Binance data | ✅ Complete |
| Sentiment analysis | ⚠️ MOCKED (API key invalid) |
| Dynamic TP/SL (ATR-based) | ✅ Complete |
| Training pipeline | ✅ Complete |
| Dual-mode training | ✅ Complete |
| User-configurable network | ✅ Complete |

## Tech Stack
- Backend: FastAPI, TensorFlow/Keras, CCXT, Transformers, pandas_ta
- Frontend: React, Recharts, TailwindCSS, Shadcn/UI
- Database: MongoDB
- ML: Bi-LSTM, Attention layers, FinBERT

## Backlog

### P0 (Critical) - None remaining for MVP

### P1 (High)
- Model persistence (save/load trained weights to disk)
- User-controlled network parameters in API (currently UI only)
- Real Twitter/Reddit API integration for sentiment

### P2 (Medium)
- Backtesting with historical data
- WebSocket for real-time price updates
- Alert system for predictions
- Time-based trade exit (15 min auto-close)

### P3 (Low)
- Full 10-year data downsampling strategy
- Performance metrics dashboard
- Multiple model comparison

## Known Issues
1. **CryptoPanic API**: Returns 404 - using MOCKED sentiment data
2. **Intermittent CORS**: Occasional "Failed to fetch" on initial load (transient)

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/health | GET | Service health check |
| /api/market/data | POST | Fetch OHLCV data |
| /api/market/latest/{symbol} | GET | Latest price |
| /api/predict | POST | Generate prediction |
| /api/training/start | POST | Start training (mode, config) |
| /api/training/status | GET | Training progress |
| /api/dashboard/stats | GET | Dashboard statistics |
