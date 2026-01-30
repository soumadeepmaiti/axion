# Omni-Crypto Hybrid Deep Learning Trading System

## Original Problem Statement
Build a Multi-Input Hybrid Deep Learning System for crypto trading using Late Fusion strategy combining high-frequency market data with real-time NLP sentiment. Features three branches: Bi-LSTM for micro data, Attention-GRU for macro trends, FinBERT+LLM for sentiment.

## User Choices
- Data Source: CCXT Binance
- Trading Pairs: BTC/USDT, ETH/USDT
- Model: Fully trainable pipeline
- Sentiment: FinBERT + Emergent LLM Key

## Architecture
- **Branch A (Micro)**: Bi-LSTM for 1m/5m/10m OHLCV data
- **Branch B (Macro)**: Attention-based GRU for 1h/1d trends
- **Branch C (Sentiment)**: FinBERT + LLM enhancement
- **Fusion**: Late fusion concatenation → Dense → Output

## What's Been Implemented (Jan 30, 2026)
- ✅ FastAPI backend with ML model infrastructure
- ✅ CCXT Binance integration for live market data
- ✅ Multi-timeframe data pipeline with technical indicators
- ✅ Hybrid model architecture (Bi-LSTM + Attention-GRU + Sentiment)
- ✅ FinBERT sentiment analysis with LLM enhancement
- ✅ Training service with progress tracking
- ✅ Prediction endpoints with ATR-based TP/SL
- ✅ React dashboard with real-time charts
- ✅ Training page with model architecture visualization
- ✅ Predictions history page
- ✅ Settings page with sentiment testing

## Core Requirements
1. Multi-input hybrid deep learning model
2. Real-time market data from Binance
3. Sentiment analysis (FinBERT + LLM)
4. Dynamic TP/SL calculation (ATR-based)
5. Training pipeline with metrics

## Tech Stack
- Backend: FastAPI, TensorFlow, CCXT, Transformers
- Frontend: React, Recharts, Tailwind, Shadcn/UI
- Database: MongoDB
- ML: Bi-LSTM, GRU with Attention, FinBERT

## Backlog
### P0 (Critical)
- None remaining for MVP

### P1 (High)
- Add real Twitter/Reddit API integration
- Implement backtesting with historical data
- Add WebSocket for real-time price updates

### P2 (Medium)
- Model persistence and loading
- Performance metrics dashboard
- Alert system for predictions
