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

### UI Layout Fix (Jan 30, 2026)
- ✅ Fixed Training.jsx layout - 4 config sections now align horizontally in single row
- Changed grid from `lg:grid-cols-5` to `lg:grid-cols-4` with proper column distribution

### Backend - Complete Advanced Trading System

#### Data Pipeline - 99 Features (191 total columns)
- ✅ **Market Microstructure** (MOCK - ready for real APIs):
  - Funding Rate (perpetual futures sentiment)
  - Open Interest changes
  - Liquidation data (long/short)
  - Exchange inflow/outflow
- ✅ **Cross-Market Signals** (MOCK - ready for real APIs):
  - S&P 500 correlation
  - DXY (Dollar Index) inverse correlation
  - Gold correlation
  - Fear & Greed Index
- ✅ **Time-Based Features**:
  - Trading sessions (Asian/EU/US)
  - Day of week, Month seasonality
  - Bitcoin halving cycle (days since/until halving)
- ✅ **Advanced Technical Analysis**:
  - Wyckoff accumulation/distribution
  - Elliott Wave patterns (swing detection, Fibonacci)
  - Ichimoku Cloud (all 5 lines, cloud signals)
  - Volume Profile (POC, VAH, VAL)
- ✅ **Whale Tracking** (MOCK - ready for real APIs):
  - Whale transaction count
  - Whale buy/sell volume
  - Whale exchange deposits/withdrawals

#### Model Architectures - 9 Types
1. **LSTM** - Bi-directional with attention
2. **GRU** - Gated Recurrent Unit
3. **Transformer** - Multi-head attention
4. **CNN + LSTM** - Convolutional + Recurrent hybrid
5. **Ensemble** - LSTM + XGBoost + RandomForest
6. **TFT** - Temporal Fusion Transformer (Google's SOTA)
7. **Multi-Task** - Predicts direction + volatility + magnitude
8. **GNN** - Graph Neural Network for asset relationships
9. **Multi-TF Attention** - Separate attention per timeframe

#### Training Features
- ✅ Walk-Forward Validation (time-series CV)
- ✅ Class Balancing (SMOTE or class weights)
- ✅ Optuna Hyperparameter Search
- ✅ Learning Rate Scheduling
- ✅ Model Persistence (save/load)

### Frontend - Complete UI
- ✅ **9 Network Types** selectable via UI
- ✅ **Advanced Options Tab** with all training features
- ✅ **Saved Models Tab** with load functionality
- ✅ **Real-time Training Progress** with charts

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

### P0 (Critical) 
- Implement Backtesting Framework (run models against historical data, generate PnL/Sharpe Ratio/Max Drawdown reports)

### P1 (High)
- Verify Model Persistence End-to-End (train → save → restart → load → predict)
- Time-based trade exit (15 min auto-close)
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
