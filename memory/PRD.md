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
- Data Source: CCXT (Multi-Exchange Support: Binance, OKX, KuCoin, Bybit, Kraken, Coinbase)
- Trading Pairs: BTC/USDT, ETH/USDT + 20 altcoins for portfolio
- Model: Fully trainable with user-configurable network depth
- Sentiment: FinBERT + CryptoPanic (MOCKED - invalid API key)
- Training Date Range: User-selectable

## Architecture
```
/app/
├── backend/
│   ├── server.py             # FastAPI endpoints
│   ├── settings.json         # User settings storage
│   ├── ml_models/
│   │   ├── hybrid_model.py
│   │   ├── advanced_models.py
│   │   ├── rl_models.py            # DQN, PPO RL agents
│   │   ├── multi_model_ensemble.py # Multi-model training
│   │   └── tcn_gnn_lstm_hybrid.py  # TCN-GNN-LSTM model
│   └── services/
│       ├── advanced_data_pipeline.py  # Multi-exchange support
│       ├── sentiment_service.py       # FinBERT (mocked)
│       ├── advanced_training_service.py
│       ├── llm_service.py             # OpenAI, Claude, Gemini
│       ├── backtesting_service.py
│       ├── correlation_analyzer.py    # NEW: Multi-asset correlations
│       ├── multi_asset_predictor.py   # NEW: ML predictions per asset
│       └── portfolio_optimizer.py     # NEW: Portfolio optimization
└── frontend/
    └── src/pages/
        ├── Dashboard.jsx   # Price charts, predictions
        ├── Training.jsx    # Network config, multi-model, RL
        ├── Portfolio.jsx   # NEW: Portfolio optimization page
        ├── Advisor.jsx     # LLM chat interface
        ├── Backtesting.jsx # Strategy backtesting
        └── Settings.jsx    # API keys, exchanges
```

## Latest Updates (Feb 2, 2026)

### Multi-Asset Portfolio Optimization - Phase 2 COMPLETE (Feb 2, 2026)
- ✅ **All 4 Strategies Now Working:**
  - **Traditional+ML**: Mean-Variance Optimization with ML-predicted returns
  - **Deep Learning**: LSTM-based neural network with attention mechanism, outputs portfolio weights via softmax
  - **RL Agent**: PPO (Proximal Policy Optimization) with actor-critic architecture
  - **Hybrid Ensemble**: Weighted average of all trained strategies
- ✅ **Model Training Endpoints:**
  - `POST /api/portfolio/train-model` with model_type='deep_learning' or 'rl_agent'
- ✅ **UI Enhancements:**
  - Header badges show "DLModel ✓" and "RLAgent ✓" when trained
  - Advanced Strategy Training section with Train DL Model and Train RL Agent buttons
  - Compare tab shows all 4 strategies with metrics
  - Hybrid card shows "Combined: traditional_ml, deep_learning, rl_agent"
- ✅ **Tested:** 100% backend (16/16 tests), 100% frontend
- ⚠️ **Note:** DL and RL models persist in memory only - need retraining after server restart

### Multi-Asset Portfolio Optimization - Phase 1 (Feb 2, 2026)
- ✅ **NEW PAGE:** `/portfolio` - Portfolio Optimizer with 4 tabs
- ✅ **20 Default Assets:** BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, FIL, APT, ARB, OP, INJ, NEAR
- ✅ **Strategy A: Traditional+ML:** Mean-Variance Optimization (Markowitz) with ML-predicted returns
- ✅ **Optimization Objectives:** Max Sharpe, Max Return, Min Risk, Risk Parity
- ✅ **Prediction Horizons:** 24h, 7d, 30d
- ✅ **Portfolio Constraints:** Max weight per asset, min assets in portfolio
- ✅ **Correlation Heatmap:** Visual correlation matrix between all assets
- ✅ **Efficient Frontier Chart:** Risk vs Return curve visualization
- ✅ **Strategy Comparison:** Side-by-side view of all 4 strategies
- ✅ **Investment Breakdown:** Shows $ allocation per asset for given investment
- ⏳ **Phase 2 Pending:** Deep Learning Portfolio, RL Agent strategies

### Backend Services Created
- `correlation_analyzer.py`: Fetches multi-asset data, calculates correlation/covariance matrices
- `multi_asset_predictor.py`: Trains LSTM models per asset, predicts returns
- `portfolio_optimizer.py`: Implements Mean-Variance Optimization with scipy

### API Endpoints Added
- `GET /api/portfolio/assets` - List available assets and strategies
- `POST /api/portfolio/fetch-data` - Fetch data for multiple assets
- `POST /api/portfolio/train` - Train ML models for all assets
- `POST /api/portfolio/optimize` - Get optimal allocation
- `GET /api/portfolio/correlation` - Get correlation matrix
- `GET /api/portfolio/efficient-frontier` - Get efficient frontier data

## Latest Updates (Feb 1, 2026)

### Arbitrary Historical Date Range Fetching (Feb 1, 2026)
- ✅ **Feature Implemented:** Users can now fetch historical data from any date (e.g., 2017 to today)
- ✅ **Chunked Fetching:** `advanced_data_pipeline.py` uses paginated API calls to avoid timeouts
- ✅ **Helper Method:** `_get_timeframe_ms()` converts timeframe strings to milliseconds
- ✅ **API Endpoints:**
  - `POST /api/training/data-preview` - Returns estimated candles for a date range
- ✅ **UI Enhancements:**
  - Calendar now shows Year dropdown (2017-current year)
  - Month dropdown for quick navigation
  - Date format: YYYY-MM-DD
  - Date range info shows "X days of historical data"
  - Large dataset warning for multi-year ranges
- ✅ **Tested:** 100% backend tests passed (10/10), all frontend tests passed
- ✅ **Verified:** Successfully trained on 3 years of daily data (1096 samples)

### Multi-Exchange Support with OKX Integration
- ✅ **6 Exchanges Supported:** Binance, OKX, KuCoin, Bybit, Kraken, Coinbase
- ✅ **API Endpoints Added:**
  - `POST /api/exchange/configure` - Configure exchange with credentials
  - `POST /api/exchange/test` - Test exchange connection
  - `POST /api/exchange/set-active` - Switch active exchange
  - `GET /api/exchange/status` - Get all exchange statuses
- ✅ **OKX Tested:** Successfully fetching BTC/USDT ticker
- ✅ **Settings UI:** Already has OKX fields (api_key, secret, passphrase)

### Dashboard Predictions Fixed
- ✅ **Auto-loads best saved model** on server startup (60.26% accuracy LSTM)
- ✅ **Uses advanced_data_pipeline** for consistent features
- ✅ **Dynamic predictions** - no more static 50% probability

## What's Been Implemented (Jan 30-31, 2026)

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

### Reinforcement Learning Architecture (Jan 31, 2026)
- ✅ **2 RL Algorithms Added:**
  - **DQN (Deep Q-Network)** - Experience replay, target network, dueling architecture
  - **PPO (Proximal Policy Optimization)** - Actor-critic, clipped surrogate objective, GAE
- ✅ **Trading Environment:** Custom TradingEnvironment class with:
  - State: Window of market features + position info
  - Actions: BUY (open long), HOLD (wait), SELL (open short)
  - Reward: PnL with transaction costs
- ✅ **RL Trainer:** Unified RLTrainer class supporting both DQN and PPO
- ✅ **UI Configuration Panel:** Training Episodes slider (50-500), Discount Factor (γ) slider (0.90-0.99)
- ✅ **Files Created:** `/app/backend/ml_models/rl_models.py`

### Multi-Model Ensemble Training (Jan 31, 2026)
- ✅ **4 Ensemble Methods:**
  - **Simple Voting** - Each model votes equally
  - **Weighted Voting** - Models weighted by validation accuracy (recommended)
  - **Stacking** - XGBoost meta-learner trained on model outputs
  - **Blending** - Optimized linear combination on holdout set
- ✅ **Supported Base Models:** LSTM, GRU, Transformer, CNN_LSTM
- ✅ **Multi-Model Benefits:**
  - Train multiple architectures simultaneously
  - Combine predictions for higher accuracy
  - View individual model performance breakdown
- ✅ **UI Configuration Panel:** Model selection checkboxes, Ensemble method dropdown
- ✅ **Files Created:** `/app/backend/ml_models/multi_model_ensemble.py`
- ✅ **Tested:** 100% (14/14 backend tests, all frontend tests passed)
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
1. **CryptoPanic API**: Returns 404 - using MOCKED sentiment data (blocked on valid API key)
2. **Intermittent CORS**: Occasional "Failed to fetch" on initial load (transient)
3. **Exchange APIs**: Data fetching uses public endpoints - user keys from Settings not yet connected

## Upcoming Tasks
1. **P1:** Connect Exchange APIs to Data Pipeline - use keys from Settings page for authenticated data
2. **P1:** End-to-End Model Persistence Test - train → save → restart → load → backtest
3. **P2:** Integrate Real On-Chain/Alternative Data APIs - replace mocked data sources

## All Supported Model Types (12 total)
| Type | Name | Description |
|------|------|-------------|
| lstm | LSTM | Long Short-Term Memory - Best for sequences |
| gru | GRU | Gated Recurrent Unit - Faster training |
| transformer | Transformer | Multi-head Attention - State of art |
| cnn_lstm | CNN + LSTM | Convolutional + Recurrent hybrid |
| ensemble | Ensemble | LSTM + XGBoost + RandomForest |
| tft | TFT | Temporal Fusion Transformer - Google's best |
| multi_task | Multi-Task | Predict direction + volatility + magnitude |
| gnn | GNN | Graph Neural Network - Asset relationships |
| multi_tf_attention | Multi-TF Attn | Separate attention per timeframe |
| rl_dqn | RL - DQN | Deep Q-Network - Learns by trading simulation |
| rl_ppo | RL - PPO | Proximal Policy Optimization - Advanced RL |
| multi_model | Multi-Model | Train multiple models & combine predictions |

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
