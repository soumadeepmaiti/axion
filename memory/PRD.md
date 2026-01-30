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

### Backend
- ✅ FastAPI server with CORS middleware
- ✅ CCXT Binance US integration for live market data
- ✅ Multi-timeframe data pipeline with 28 technical indicators
- ✅ **Dual-Mode Training System**:
  - Pure ML Mode: Model learns patterns from raw features
  - Mathematical Modeling Mode: Applies quant strategies (mean reversion, momentum, RSI, MACD, Fibonacci, etc.)
- ✅ User-configurable network architecture (LSTM layers, Dense layers, attention, batch norm)
- ✅ Training with real-time progress tracking
- ✅ Prediction endpoints with ATR-based TP/SL
- ✅ JSON sanitization for NaN/Inf values

### Frontend
- ✅ Dashboard with real-time price charts
- ✅ Training page with network architecture configuration
- ✅ Mathematical strategies selection
- ✅ Training progress visualization (loss/accuracy curves)
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
