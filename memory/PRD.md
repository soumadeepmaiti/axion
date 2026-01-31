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

### Settings Page Reorganization (Jan 31, 2026)
- ✅ Merged Exchanges into "Data & Exchanges" tab with sub-tabs:
  - **Exchanges**: Binance, Coinbase, KuCoin, Kraken, Bybit, OKX
  - **Sentiment APIs**: CryptoPanic, Twitter, Reddit
  - **Market Data**: Glassnode, Alpha Vantage
  - **Custom APIs**: 3 custom API slots
- ✅ Fixed LLM chat/signal endpoints (corrected `get_latest_data` method call)
- ✅ 5 main tabs: Data & Exchanges, LLM Models, Trading, Alerts, Display

### LLM Multi-Model Integration (Jan 31, 2026)
- ✅ **3 LLM Providers Active:** OpenAI (gpt-4o), Claude (claude-4-sonnet), Gemini (gemini-2.0-flash)
- ✅ **Emergent LLM Key Integration:** Using `emergentintegrations` library with universal API key
- ✅ **Backend LLM Service:** `/app/backend/services/llm_service.py` - fully refactored to use LlmChat class
- ✅ **3 LLM Features:**
  - **Sentiment Analysis** - LLM analyzes news → sentiment score as ML feature
  - **Ensemble Voting** - Multiple LLMs vote BUY/SELL/HOLD with consensus (100% tested)
  - **Chat Advisor** - Direct chat interface for market questions (all 3 providers working)
- ✅ **AI Advisor Page:** New `/advisor` route with Chat + Ensemble Signal tabs
- ✅ **"3 LLMs Active" Badge:** Shows in UI when all providers are working
- ✅ Added API endpoints: GET/POST /api/llm/chat, /api/llm/signal, /api/llm/sentiment, /api/llm/multi-chat, /api/llm/providers
- ✅ **Tested:** 100% backend tests passed (8/8), frontend fully functional

### Settings Page Overhaul (Jan 31, 2026)
- ✅ **API Keys Management:** CryptoPanic, Twitter, Reddit, Glassnode, Alpha Vantage
- ✅ **Trading Settings:** Stop-loss, take-profit, position size, risk per trade, trailing stop
- ✅ **Notification Settings:** Email alerts, price alerts with add/remove functionality
- ✅ **Data Source Settings:** Exchange selection, default symbol/timeframe, trading pairs
- ✅ **Theme/Display Settings:** Dark/Light/System mode, chart style preferences
- ✅ **Model Management:** View all saved models, load/delete models, see active model
- ✅ Added backend endpoints: GET/POST /api/settings, DELETE /api/models/{path}

### Backtesting Framework (Jan 30, 2026)
- ✅ Created `/app/backend/services/backtesting_service.py` - Full backtesting simulation engine
- ✅ Added API endpoints: POST /api/backtest/start, GET /api/backtest/status, GET /api/backtest/result, POST /api/backtest/stop, GET /api/backtest/history
- ✅ Created `/app/frontend/src/pages/Backtesting.jsx` - Comprehensive UI with 3 tabs (Configuration, Results, History)
- ✅ Performance metrics: Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate, Profit Factor
- ✅ Charts: Equity Curve, Drawdown Curve, Monthly Returns
- ✅ Trade table with entry/exit details
- ✅ Model validation before backtest starts

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
| Backtesting Framework | ✅ Complete |

## Tech Stack
- Backend: FastAPI, TensorFlow/Keras, CCXT, Transformers, ta
- Frontend: React, Recharts, TailwindCSS, Shadcn/UI
- Database: MongoDB
- ML: Bi-LSTM, Attention layers, FinBERT, XGBoost

## Backlog

### P0 (Critical) 
- ~~Implement Backtesting Framework~~ ✅ COMPLETED

### P1 (High)
- Verify Model Persistence End-to-End (train → save → restart → load → predict)
- Time-based trade exit (15 min auto-close)
- Real Twitter/Reddit API integration for sentiment

### P2 (Medium)
- WebSocket for real-time price updates
- Alert system for predictions

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
| /api/training/advanced/start | POST | Start advanced training |
| /api/training/status | GET | Training progress |
| /api/training/advanced/status | GET | Advanced training progress |
| /api/models/saved | GET | List saved models |
| /api/models/load | POST | Load a saved model |
| /api/backtest/start | POST | Start backtest (requires loaded model) |
| /api/backtest/status | GET | Get backtest progress |
| /api/backtest/result | GET | Get backtest results |
| /api/backtest/stop | POST | Stop running backtest |
| /api/backtest/history | GET | Get backtest history |
| /api/dashboard/stats | GET | Dashboard statistics |
