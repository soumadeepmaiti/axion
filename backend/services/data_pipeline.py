"""
Data Pipeline Service for fetching and preprocessing cryptocurrency data
Uses CCXT for Binance exchange data
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import RobustScaler
import ta
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles all data fetching and preprocessing for the hybrid model"""
    
    TIMEFRAMES = {
        'micro': ['1m', '5m'],
        'macro': ['1h', '1d']
    }
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.scalers: Dict[str, RobustScaler] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using ta library"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Trend Indicators
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # Distance from EMAs (normalized)
        df['dist_ema_9'] = (df['close'] - df['ema_9']) / df['ema_9'] * 100
        df['dist_ema_21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
        df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        
        # Volatility Indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close'] * 100
        
        # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price changes
        df['returns_1'] = df['close'].pct_change(1) * 100
        df['returns_5'] = df['close'].pct_change(5) * 100
        df['returns_10'] = df['close'].pct_change(10) * 100
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        
        return df
    
    def prepare_micro_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for the micro (execution) branch"""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'stoch_k', 'stoch_d',
            'atr', 'bb_width', 'volume_ratio',
            'returns_1', 'returns_5', 'body_size'
        ]
        
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        features = df[feature_cols].fillna(0).values
        return features
    
    def prepare_macro_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for the macro (context) branch - technical embeddings"""
        feature_cols = [
            'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'dist_ema_200',
            'rsi', 'macd', 'bb_width',
            'returns_5', 'returns_10', 'volume_ratio'
        ]
        
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        features = df[feature_cols].fillna(0).values
        return features
    
    def scale_features(self, features: np.ndarray, key: str, fit: bool = False) -> np.ndarray:
        """Scale features using RobustScaler"""
        if key not in self.scalers:
            self.scalers[key] = RobustScaler()
            fit = True
            
        if fit:
            return self.scalers[key].fit_transform(features)
        return self.scalers[key].transform(features)
    
    async def get_training_data(self, symbol: str, lookback_days: int = 30) -> Dict:
        """Get preprocessed data for model training"""
        # Fetch data for different timeframes
        micro_1m = await self.fetch_ohlcv(symbol, '1m', limit=min(lookback_days * 1440, 1000))
        micro_5m = await self.fetch_ohlcv(symbol, '5m', limit=min(lookback_days * 288, 1000))
        macro_1h = await self.fetch_ohlcv(symbol, '1h', limit=min(lookback_days * 24, 1000))
        macro_1d = await self.fetch_ohlcv(symbol, '1d', limit=min(lookback_days, 365))
        
        # Calculate indicators
        micro_1m = self.calculate_technical_indicators(micro_1m)
        micro_5m = self.calculate_technical_indicators(micro_5m)
        macro_1h = self.calculate_technical_indicators(macro_1h)
        macro_1d = self.calculate_technical_indicators(macro_1d)
        
        # Prepare features
        micro_features = self.prepare_micro_features(micro_5m)
        macro_features = self.prepare_macro_features(macro_1h)
        
        # Scale features
        if len(micro_features) > 0:
            micro_features = self.scale_features(micro_features, f'{symbol}_micro', fit=True)
        if len(macro_features) > 0:
            macro_features = self.scale_features(macro_features, f'{symbol}_macro', fit=True)
        
        # Create labels (price direction after 5 minutes)
        if not micro_5m.empty:
            micro_5m['future_return'] = micro_5m['close'].shift(-1) / micro_5m['close'] - 1
            micro_5m['label'] = (micro_5m['future_return'] > 0).astype(int)
            labels = micro_5m['label'].dropna().values
        else:
            labels = np.array([])
        
        return {
            'micro_features': micro_features[:-1] if len(micro_features) > 1 else micro_features,
            'macro_features': macro_features,
            'labels': labels,
            'raw_data': {
                '1m': micro_1m.reset_index().to_dict('records') if not micro_1m.empty else [],
                '5m': micro_5m.reset_index().to_dict('records') if not micro_5m.empty else [],
                '1h': macro_1h.reset_index().to_dict('records') if not macro_1h.empty else [],
                '1d': macro_1d.reset_index().to_dict('records') if not macro_1d.empty else []
            }
        }
    
    async def get_latest_data(self, symbol: str) -> Dict:
        """Get latest data for real-time prediction"""
        micro_5m = await self.fetch_ohlcv(symbol, '5m', limit=100)
        macro_1h = await self.fetch_ohlcv(symbol, '1h', limit=100)
        
        micro_5m = self.calculate_technical_indicators(micro_5m)
        macro_1h = self.calculate_technical_indicators(macro_1h)
        
        micro_features = self.prepare_micro_features(micro_5m)
        macro_features = self.prepare_macro_features(macro_1h)
        
        if f'{symbol}_micro' in self.scalers and len(micro_features) > 0:
            micro_features = self.scale_features(micro_features, f'{symbol}_micro', fit=False)
        if f'{symbol}_macro' in self.scalers and len(macro_features) > 0:
            macro_features = self.scale_features(macro_features, f'{symbol}_macro', fit=False)
        
        current_price = float(micro_5m['close'].iloc[-1]) if not micro_5m.empty else 0
        current_atr = float(micro_5m['atr'].iloc[-1]) if not micro_5m.empty and 'atr' in micro_5m.columns else 0
        
        return {
            'micro_features': micro_features[-50:] if len(micro_features) > 50 else micro_features,
            'macro_features': macro_features[-24:] if len(macro_features) > 24 else macro_features,
            'current_price': current_price,
            'current_atr': current_atr,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def calculate_tp_sl(self, current_price: float, atr: float, direction: int) -> Dict:
        """Calculate Take Profit and Stop Loss based on ATR"""
        if direction == 1:  # Long
            tp = current_price + (2.5 * atr)
            sl = current_price - (1.5 * atr)
        else:  # Short
            tp = current_price - (2.5 * atr)
            sl = current_price + (1.5 * atr)
            
        return {
            'take_profit': round(tp, 2),
            'stop_loss': round(sl, 2),
            'risk_reward': 2.5 / 1.5
        }


# Singleton instance
data_pipeline = DataPipeline()
