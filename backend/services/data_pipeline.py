"""
Pure ML Data Pipeline - No Imposed Mathematical Conditions
Data patterns are learned by the model, not predefined
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

logger = logging.getLogger(__name__)


class PureMLDataPipeline:
    """Data pipeline that lets ML model discover patterns - no imposed formulas"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.exchange = ccxt.binanceus({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.scalers: Dict[str, RobustScaler] = {}
        
    async def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h',
        start_date: str = None,
        end_date: str = None,
        max_candles: int = 10000
    ) -> pd.DataFrame:
        """Fetch historical data within user-specified date range"""
        all_data = []
        
        # Parse dates
        if start_date:
            since = int(datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp() * 1000)
        else:
            since = None
            
        if end_date:
            until = int(datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp() * 1000)
        else:
            until = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        logger.info(f"Fetching {symbol} {timeframe} from {start_date} to {end_date}")
        
        loop = asyncio.get_event_loop()
        
        while len(all_data) < max_candles:
            try:
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                )
                
                if not ohlcv:
                    break
                
                # Filter by end date
                ohlcv = [candle for candle in ohlcv if candle[0] <= until]
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                
                # Move to next batch
                since = ohlcv[-1][0] + 1
                
                # Stop if we've passed the end date
                if since > until:
                    break
                
                await asyncio.sleep(0.1)
                
                if len(ohlcv) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch recent OHLCV data"""
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
    
    def calculate_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate raw technical indicators - model will learn which matter"""
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        
        # Price-based features (raw, no interpretation)
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Volatility features
        df['volatility_10'] = df['return_1'].rolling(10).std()
        df['volatility_20'] = df['return_1'].rolling(20).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change(1)
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Price relative to moving averages (let model interpret)
        for period in [5, 10, 20, 50, 100]:
            ma = df['close'].rolling(period).mean()
            df[f'price_ma{period}_ratio'] = df['close'] / ma
        
        # Momentum indicators (raw values, no thresholds)
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_ratio'] = df['atr'] / df['close']
        
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Candle features
        df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        return df
    
    # Alias for backward compatibility
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for calculate_raw_features"""
        return self.calculate_raw_features(df)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for ML - all features, model decides importance"""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1', 'return_5', 'return_10', 'return_20',
            'volatility_10', 'volatility_20',
            'volume_change', 'volume_ma_ratio',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio', 'price_ma50_ratio', 'price_ma100_ratio',
            'rsi_14', 'rsi_7', 'macd', 'stoch',
            'atr_ratio', 'bb_position', 'bb_width',
            'body_ratio', 'upper_shadow', 'lower_shadow',
            'adx'
        ]
        
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        features = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        return features
    
    def scale_features(self, features: np.ndarray, key: str, fit: bool = False) -> np.ndarray:
        """Scale features using RobustScaler"""
        if key not in self.scalers:
            self.scalers[key] = RobustScaler()
            fit = True
            
        if fit:
            return self.scalers[key].fit_transform(features)
        return self.scalers[key].transform(features)
    
    async def get_training_data(
        self, 
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        timeframe: str = '1h'
    ) -> Dict:
        """Get training data for user-specified date range"""
        logger.info(f"Fetching training data: {symbol} from {start_date} to {end_date}")
        
        # Fetch data for the specified range
        df = await self.fetch_historical_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            return {"error": "Could not fetch historical data for the specified range"}
        
        # Calculate features
        df = self.calculate_raw_features(df)
        
        # Prepare features
        features = self.prepare_features(df)
        features = self.scale_features(features, f'{symbol}_train', fit=True)
        
        # Create labels (next candle direction)
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)
        labels = df['label'].dropna().values
        
        # Align features and labels
        min_len = min(len(features) - 1, len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"Prepared {len(features)} samples for training")
        
        return {
            'features': features,
            'labels': labels,
            'prices': df['close'].values,  # For mathematical modeling
            'data_info': {
                'total_candles': len(df),
                'training_samples': len(features),
                'timeframe': timeframe,
                'date_range': {
                    'start': str(df.index[0]) if len(df) > 0 else None,
                    'end': str(df.index[-1]) if len(df) > 0 else None
                },
                'feature_count': features.shape[1] if len(features) > 0 else 0
            }
        }
    
    async def get_latest_data(self, symbol: str) -> Dict:
        """Get latest data for prediction"""
        df = await self.fetch_ohlcv(symbol, '1h', limit=150)
        
        if df.empty:
            return {
                'features': np.array([]),
                'current_price': 0.0,
                'current_atr': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': 'No data available'
            }
        
        df = self.calculate_raw_features(df)
        features = self.prepare_features(df)
        
        if f'{symbol}_train' in self.scalers:
            features = self.scale_features(features, f'{symbol}_train', fit=False)
        
        current_price = float(df['close'].iloc[-1]) if not df.empty else 0.0
        current_atr = float(df['atr'].iloc[-1]) if not df.empty and 'atr' in df.columns else 0.0
        
        # Handle NaN/Inf values
        if np.isnan(current_price) or np.isinf(current_price):
            current_price = 0.0
        if np.isnan(current_atr) or np.isinf(current_atr):
            current_atr = current_price * 0.02 if current_price > 0 else 0.0
        
        return {
            'features': features[-100:] if len(features) > 100 else features,
            'current_price': current_price,
            'current_atr': current_atr,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def calculate_tp_sl(self, current_price: float, atr: float, direction: int, math_analysis: Dict = None) -> Dict:
        """Calculate TP/SL based on ATR (volatility-adjusted)"""
        if atr == 0 or np.isnan(atr) or np.isinf(atr):
            atr = current_price * 0.02  # Default 2% if ATR not available
        
        # Handle invalid current_price
        if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
            return {
                'take_profit': 0.0,
                'stop_loss': 0.0,
                'risk_reward': 1.67
            }
            
        if direction == 1:  # Long
            tp = current_price + (2.5 * atr)
            sl = current_price - (1.5 * atr)
        else:  # Short
            tp = current_price - (2.5 * atr)
            sl = current_price + (1.5 * atr)
        
        return {
            'take_profit': round(float(tp), 2),
            'stop_loss': round(float(sl), 2),
            'risk_reward': round(2.5 / 1.5, 2)
        }


# Singleton instance
data_pipeline = PureMLDataPipeline()
