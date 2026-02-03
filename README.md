<![CDATA[<div align="center">

# ⚡ AXION

### *Allocate with Intelligence*

**A Next-Generation Multi-Asset Portfolio Optimization Platform Powered by Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

---

</div>

## 🎯 Overview

**Axion** is an advanced cryptocurrency portfolio optimization system that combines cutting-edge deep learning architectures with traditional quantitative finance techniques. The platform features a novel **TCN-GNN-LSTM** hybrid model that captures multi-scale temporal patterns, dynamic cross-asset correlations, and provides uncertainty-aware predictions.

### Why Axion?

| Traditional Approaches | Axion's Solution |
|----------------------|------------------|
| Single-scale features | **TCN** with dilated convolutions captures 1m, 1h, 1d, 1w patterns |
| Static correlation matrices | **GNN** learns dynamic, time-varying asset relationships |
| Point predictions only | **Gaussian head** provides uncertainty quantification |
| One-size-fits-all strategy | **4 AI strategies** to match different risk profiles |

---

## ✨ Features

### 🏠 Trading Dashboard
Real-time market monitoring with live price feeds, sentiment analysis, volume tracking, and AI-powered predictions.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/dashboard.png" alt="Dashboard" width="90%">
</p>

**Key Features:**
- Live price charts with multiple timeframes (1m, 5m, 1h, 1d)
- Real-time sentiment analysis from multiple sources
- Volume analysis and market depth visualization
- One-click AI prediction generation

---

### 🧠 Advanced Model Training
Train state-of-the-art deep learning models with a visual, no-code interface.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/training.png" alt="Training" width="90%">
</p>

**Supported Architectures:**
| Model | Description | Best For |
|-------|-------------|----------|
| **LSTM** | Long Short-Term Memory | Sequential patterns |
| **GRU** | Gated Recurrent Unit | Faster training |
| **CNN + LSTM** | Convolutional + Recurrent | Local + temporal patterns |
| **TCN-GNN-LSTM** | Our novel hybrid | Multi-scale + cross-asset |
| **Transformer** | Multi-head attention | Long-range dependencies |
| **TFT** | Temporal Fusion Transformer | Interpretable forecasting |
| **RL-DQN** | Deep Q-Network | Adaptive strategies |
| **RL-PPO** | Proximal Policy Optimization | Stable RL training |

**Training Features:**
- Configurable hyperparameters (layers, units, dropout, learning rate)
- Visual architecture preview
- Real-time training progress with loss/accuracy curves
- Model persistence and versioning

---

### 📊 Portfolio Optimizer
Multi-asset portfolio optimization with 4 distinct AI strategies.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/portfolio.png" alt="Portfolio" width="90%">
</p>

**Optimization Strategies:**

1. **Traditional + ML**
   - Combines Markowitz mean-variance optimization with ML predictions
   - Uses XGBoost for return forecasting
   - Best for: Conservative investors

2. **Deep Learning**
   - Pure deep learning approach using TCN-GNN-LSTM
   - Captures complex non-linear patterns
   - Best for: Tech-forward traders

3. **Reinforcement Learning**
   - PPO-based agent learns optimal allocation
   - Adapts to changing market conditions
   - Best for: Dynamic market environments

4. **Hybrid Ensemble**
   - Combines all strategies with confidence weighting
   - Uses uncertainty for risk management
   - Best for: Balanced approach

**Configuration Options:**
- Select from 20+ cryptocurrencies
- Adjustable investment amount
- Prediction horizon (1-30 days)
- Max weight constraints per asset
- Multiple optimization objectives (Sharpe, Sortino, Max Return)

---

### 📈 Backtesting Engine
Comprehensive strategy backtesting with detailed performance analytics.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/backtesting.png" alt="Backtesting" width="90%">
</p>

**Features:**
- Historical data simulation
- Risk management settings (stop-loss, take-profit)
- Position sizing controls
- Commission modeling
- Performance metrics (Sharpe, Sortino, Max Drawdown, Win Rate)
- Equity curve visualization
- Trade-by-trade analysis

---

### 🤖 AI Trading Advisor
Interactive AI assistant powered by LLMs for market analysis and trading guidance.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/advisor.png" alt="AI Advisor" width="90%">
</p>

**Capabilities:**
- Real-time market analysis
- Technical indicator explanations
- Trading strategy recommendations
- Risk assessment
- Multi-LLM comparison mode (OpenAI, Anthropic, Gemini)

**Sample Queries:**
- "What's your BTC outlook for the next week?"
- "Explain RSI divergence and its implications"
- "Best entry point for ETH given current conditions?"

---

### 📉 Prediction History
Track and analyze all AI predictions with confidence metrics.

<p align="center">
  <img src="https://raw.githubusercontent.com/Soumadeep21/axion/main/assets/predictions.png" alt="Predictions" width="90%">
</p>

**Tracked Metrics:**
- Prediction direction (LONG/SHORT)
- Probability score
- Confidence level
- Entry price, Take profit, Stop loss
- Historical accuracy by timeframe

---

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AXION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐  │
│  │   React     │────▶│   FastAPI    │────▶│      MongoDB            │  │
│  │  Frontend   │◀────│   Backend    │◀────│     Database            │  │
│  └─────────────┘     └──────────────┘     └─────────────────────────┘  │
│         │                   │                                           │
│         │                   ▼                                           │
│         │            ┌──────────────┐                                   │
│         │            │   Services   │                                   │
│         │            ├──────────────┤                                   │
│         │            │ • Data Pipeline (CCXT)                           │
│         │            │ • Training Service                               │
│         │            │ • Portfolio Optimizer                            │
│         │            │ • Backtesting Engine                             │
│         │            │ • LLM Service                                    │
│         │            │ • Sentiment Analyzer                             │
│         │            └──────────────┘                                   │
│         │                   │                                           │
│         │                   ▼                                           │
│         │            ┌──────────────┐                                   │
│         │            │  ML Models   │                                   │
│         │            ├──────────────┤                                   │
│         │            │ • LSTM / GRU                                     │
│         │            │ • CNN + LSTM                                     │
│         │            │ • TCN-GNN-LSTM (Novel)                           │
│         │            │ • Transformer / TFT                              │
│         │            │ • RL (DQN, PPO)                                  │
│         │            └──────────────┘                                   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    EXTERNAL SERVICES                             │   │
│  │  • Binance/Coinbase/Kraken (via CCXT)                           │   │
│  │  • OpenAI / Anthropic / Google (LLM Providers)                  │   │
│  │  • News APIs (Sentiment)                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### TCN-GNN-LSTM Model Architecture

Our novel hybrid architecture combines three powerful deep learning paradigms:

```
                    RAW MARKET DATA (OHLCV + Features)
                                 │
                                 ▼
            ┌────────────────────────────────────────┐
            │       TCN FEATURE EXTRACTOR            │
            │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐   │
            │  │ d=1 │  │ d=2 │  │ d=4 │  │ d=8 │   │
            │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘   │
            │     └────────┴────────┴────────┘       │
            │         Multi-Scale Features           │
            └────────────────────────────────────────┘
                                 │
                                 ▼
            ┌────────────────────────────────────────┐
            │       GRAPH NEURAL NETWORK             │
            │                                        │
            │       BTC ─────── ETH                  │
            │        │  \     /  │                   │
            │        │   \   /   │   Dynamic         │
            │        │    \ /    │   Attention       │
            │        │     X     │   Weights         │
            │        │    / \    │                   │
            │        │   /   \   │                   │
            │       SOL ─────── BNB                  │
            │                                        │
            │      Cross-Asset Relationships         │
            └────────────────────────────────────────┘
                                 │
                                 ▼
            ┌────────────────────────────────────────┐
            │       LSTM PROCESSOR                   │
            │   Bidirectional + Temporal Attention   │
            │   h(t-2) ──▶ h(t-1) ──▶ h(t)          │
            └────────────────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌───────────┐  ┌───────────┐  ┌───────────┐
          │  TRADING  │  │ PREDICTION│  │   VALUE   │
          │   HEAD    │  │   HEAD    │  │   HEAD    │
          │  softmax  │  │  μ, σ     │  │  V(s)     │
          └───────────┘  └───────────┘  └───────────┘
                 │               │               │
                 ▼               ▼               ▼
          [Portfolio]    [Uncertainty]    [RL Value]
           Weights        Estimates        Function
```

**Key Innovations:**

1. **Multi-Scale Temporal Features (TCN)**
   - Dilated causal convolutions with rates [1, 2, 4, 8]
   - Captures patterns from minutes to weeks
   - Fully parallelizable (faster than LSTM alone)

2. **Dynamic Cross-Asset Modeling (GNN)**
   - Graph attention learns time-varying correlations
   - Adapts to market regime changes
   - Interpretable attention weights

3. **Uncertainty Quantification**
   - Gaussian prediction head outputs μ and σ
   - Confidence-weighted portfolio allocation
   - Risk management through uncertainty

4. **Multi-Task Learning**
   - Three output heads prevent overfitting
   - Prediction head acts as regularizer
   - Optional RL fine-tuning

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB 6.0+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/axion.git
cd axion

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
# or
yarn install

# Start MongoDB (if not running)
mongod --dbpath /path/to/data

# Start the backend
cd ../backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Start the frontend (new terminal)
cd ../frontend
npm start
```

### Environment Variables

Create `.env` files:

**backend/.env**
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=axion
OPENAI_API_KEY=your_key_here  # Optional, for AI Advisor
```

**frontend/.env**
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 📖 Usage

### 1. Dashboard

Navigate to the Dashboard to view real-time market data:
- Select a trading pair (BTC/USDT, ETH/USDT, etc.)
- View live price charts and volume analysis
- Generate AI predictions with one click

### 2. Training Models

1. Go to **Training** page
2. Select data source and trading pair
3. Choose architecture (LSTM, TCN-GNN-LSTM, etc.)
4. Configure hyperparameters
5. Click **Start Training**
6. Monitor progress in real-time
7. Save trained models for later use

### 3. Portfolio Optimization

1. Navigate to **Portfolio**
2. Select assets (up to 20)
3. Configure investment settings
4. Choose optimization strategy
5. Click **Optimize Portfolio**
6. Review allocation and metrics
7. Compare strategies side-by-side

### 4. Backtesting

1. Go to **Backtesting**
2. Configure test parameters
3. Set risk management rules
4. Run backtest
5. Analyze performance metrics
6. Review trade history

---

## 📁 Project Structure

```
axion/
├── backend/
│   ├── server.py                 # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── settings.json             # App configuration
│   ├── services/
│   │   ├── data_pipeline.py      # CCXT data fetching
│   │   ├── training_service.py   # Model training
│   │   ├── portfolio_optimizer.py # Portfolio optimization
│   │   ├── backtesting_service.py # Backtesting engine
│   │   ├── llm_service.py        # LLM integration
│   │   └── sentiment_service.py  # Sentiment analysis
│   ├── ml_models/
│   │   ├── tcn_gnn_lstm_hybrid.py # Novel architecture
│   │   ├── hybrid_model.py       # CNN+LSTM models
│   │   ├── rl_models.py          # RL agents
│   │   └── advanced_architectures.py
│   ├── saved_models/             # Persisted models
│   └── tests/                    # API tests
│
├── frontend/
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   └── App.jsx               # Main app
│   ├── public/
│   └── package.json
│
├── assets/                       # Screenshots & images
├── notebooks/                    # Jupyter notebooks
│   └── tcn_gnn_lstm_architecture_proposal.ipynb
└── README.md
```

---

## 📊 Performance

### Backtest Results (Sample)

| Strategy | Sharpe Ratio | Max Drawdown | Annual Return |
|----------|--------------|--------------|---------------|
| Traditional + ML | 1.2 | -15.3% | 45.2% |
| Deep Learning | 1.5 | -12.8% | 62.1% |
| Reinforcement Learning | 1.3 | -18.2% | 55.7% |
| **Hybrid Ensemble** | **1.7** | **-10.5%** | **71.3%** |

*Results from backtesting on 2023-2024 data. Past performance does not guarantee future results.*

---

## 🔬 Research

The TCN-GNN-LSTM architecture is documented in detail in our Jupyter notebook:

📓 **[TCN-GNN-LSTM Architecture Proposal](backend/tcn_gnn_lstm_architecture_proposal.ipynb)**

This notebook includes:
- Complete theoretical foundation
- Mathematical formulations
- Implementation code
- Training strategies (curriculum learning)
- Uncertainty-aware prediction pipeline
- Evaluation and visualization

---

## 🛠 Tech Stack

### Backend
- **FastAPI** - High-performance async API framework
- **TensorFlow/Keras** - Deep learning models
- **PyPortfolioOpt** - Portfolio optimization
- **CCXT** - Cryptocurrency exchange integration
- **scikit-learn, XGBoost** - Traditional ML
- **stable-baselines3** - Reinforcement learning

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Shadcn/UI** - Component library

### Database
- **MongoDB** - Document storage for models and predictions

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** 

- Not financial advice
- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always do your own research before investing

---

## 📬 Contact

- **Author:** [Soumadeep21](https://github.com/Soumadeep21)
- **Project Link:** [https://github.com/Soumadeep21/axion](https://github.com/Soumadeep21/axion)
- **Issues:** [GitHub Issues](https://github.com/Soumadeep21/axion/issues)

---

<div align="center">

**Built with ❤️ by the Axion Team**

*Allocate with Intelligence*

</div>
]]>