"""
Enhanced Data Pipeline with Full Historical Data and Mathematical Modeling
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

class MathematicalModeling:
    """Mathematical pattern recognition for crypto trading"""
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict:
        """Find support and resistance levels using pivot points"""
        if len(df) < window:
            return {"support": [], "resistance": []}
        
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find local maxima (resistance) and minima (support)
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(float(df['high'].iloc[i]))
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(float(df['low'].iloc[i]))
        
        # Cluster similar levels
        resistance_levels = MathematicalModeling._cluster_levels(resistance_levels)
        support_levels = MathematicalModeling._cluster_levels(support_levels)
        
        return {
            "support": support_levels[-5:] if support_levels else [],
            "resistance": resistance_levels[-5:] if resistance_levels else []
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster similar price levels together"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
            else:
                clustered[-1] = (clustered[-1] + level) / 2
        
        return clustered
    
    @staticmethod
    def fibonacci_retracement(df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if len(df) < lookback:
            lookback = len(df)
        
        recent = df.tail(lookback)
        high = float(recent['high'].max())
        low = float(recent['low'].min())
        diff = high - low
        
        fib_levels = {
            "0.0": high,
            "0.236": high - (diff * 0.236),
            "0.382": high - (diff * 0.382),
            "0.5": high - (diff * 0.5),
            "0.618": high - (diff * 0.618),
            "0.786": high - (diff * 0.786),
            "1.0": low
        }
        
        return {
            "levels": fib_levels,
            "high": high,
            "low": low,
            "trend": "bullish" if recent['close'].iloc[-1] > recent['close'].iloc[0] else "bearish"
        }
    
    @staticmethod
    def detect_trend(df: pd.DataFrame) -> Dict:
        """Detect market trend using multiple methods"""
        if len(df) < 50:
            return {"trend": "neutral", "strength": 0}
        
        close = df['close'].values
        
        # Linear regression trend
        x = np.arange(len(close))
        slope, intercept = np.polyfit(x, close, 1)
        
        # Normalize slope as percentage
        slope_pct = (slope / close.mean()) * 100
        
        # EMA trend
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        ema_trend = "bullish" if ema_20.iloc[-1] > ema_50.iloc[-1] else "bearish"
        
        # ADX for trend strength
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        trend_strength = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
        
        # Determine overall trend
        if slope_pct > 0.1 and ema_trend == "bullish":
            trend = "strong_bullish"
        elif slope_pct > 0 and ema_trend == "bullish":
            trend = "bullish"
        elif slope_pct < -0.1 and ema_trend == "bearish":
            trend = "strong_bearish"
        elif slope_pct < 0 and ema_trend == "bearish":
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "slope_pct": round(slope_pct, 4),
            "ema_trend": ema_trend,
            "adx_strength": round(trend_strength, 2),
            "regression_line": {"slope": float(slope), "intercept": float(intercept)}
        }
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick and chart patterns"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        # Doji detection
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        doji = body < (range_hl * 0.1)
        if doji.iloc[-1]:
            patterns.append({"name": "doji", "type": "reversal", "position": len(df) - 1})
        
        # Hammer detection
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        hammer = (lower_wick > body * 2) & (upper_wick < body * 0.5)
        if hammer.iloc[-1]:
            patterns.append({"name": "hammer", "type": "bullish_reversal", "position": len(df) - 1})
        
        # Engulfing pattern
        if len(df) >= 2:
            prev_body = df['close'].iloc[-2] - df['open'].iloc[-2]
            curr_body = df['close'].iloc[-1] - df['open'].iloc[-1]
            
            if prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body):
                patterns.append({"name": "bullish_engulfing", "type": "bullish_reversal", "position": len(df) - 1})
            elif prev_body > 0 and curr_body < 0 and abs(curr_body) > abs(prev_body):
                patterns.append({"name": "bearish_engulfing", "type": "bearish_reversal", "position": len(df) - 1})
        
        # Double top/bottom detection (simplified)
        if len(df) >= 50:
            highs = df['high'].tail(50)
            lows = df['low'].tail(50)
            
            high_peaks = highs[highs == highs.rolling(5, center=True).max()]
            low_peaks = lows[lows == lows.rolling(5, center=True).min()]
            
            if len(high_peaks) >= 2:
                last_two_highs = high_peaks.tail(2).values
                if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] < 0.02:
                    patterns.append({"name": "double_top", "type": "bearish_reversal", "level": float(last_two_highs.mean())})
            
            if len(low_peaks) >= 2:
                last_two_lows = low_peaks.tail(2).values
                if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] < 0.02:
                    patterns.append({"name": "double_bottom", "type": "bullish_reversal", "level": float(last_two_lows.mean())})
        
        return patterns
    
    @staticmethod
    def calculate_volatility_regime(df: pd.DataFrame) -> Dict:
        """Determine volatility regime"""
        if len(df) < 20:
            return {"regime": "unknown", "current_vol": 0, "avg_vol": 0}
        
        returns = df['close'].pct_change().dropna()
        current_vol = float(returns.tail(20).std() * np.sqrt(252) * 100)
        avg_vol = float(returns.std() * np.sqrt(252) * 100)
        
        if current_vol > avg_vol * 1.5:
            regime = "high_volatility"
        elif current_vol < avg_vol * 0.5:
            regime = "low_volatility"
        else:
            regime = "normal"
        
        return {
            "regime": regime,
            "current_vol": round(current_vol, 2),
            "avg_vol": round(avg_vol, 2),
            "vol_ratio": round(current_vol / avg_vol if avg_vol > 0 else 1, 2)
        }


class EnhancedDataPipeline:
    """Enhanced data pipeline with full historical data fetching"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        # Use binanceus for unrestricted access
        self.exchange = ccxt.binanceus({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.scalers: Dict[str, RobustScaler] = {}
        self.math_model = MathematicalModeling()
        
    async def fetch_all_historical_data(self, symbol: str, timeframe: str = '1h', max_candles: int = 10000) -> pd.DataFrame:
        """Fetch maximum available historical data by paginating through API"""
        all_data = []
        since = None
        
        logger.info(f"Fetching all historical data for {symbol} {timeframe}...")
        
        loop = asyncio.get_event_loop()
        
        while len(all_data) < max_candles:
            try:
                ohlcv = await loop.run_in_executor(
                    None,
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Move to next batch
                since = ohlcv[-1][0] + 1
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
                if len(ohlcv) < 1000:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df
    
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
            
            logger.info(f"Fetched {len(df)} rows from binanceus")
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        
        # Trend Indicators
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Distance from EMAs (normalized)
        df['dist_ema_9'] = (df['close'] - df['ema_9']) / df['ema_9'] * 100
        df['dist_ema_21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
        df['dist_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_6'] = ta.momentum.rsi(df['close'], window=6)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # Volatility Indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'], window=20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # Price changes at multiple timeframes
        for period in [1, 3, 5, 10, 20, 50]:
            df[f'returns_{period}'] = df['close'].pct_change(period) * 100
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        df['candle_direction'] = (df['close'] > df['open']).astype(int)
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['plus_di'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df['minus_di'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare all features for model training"""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'rsi_6', 'macd', 'stoch_k', 'stoch_d', 'cci', 'mfi',
            'atr_pct', 'bb_width', 'bb_pct', 'volume_ratio',
            'returns_1', 'returns_3', 'returns_5', 'returns_10',
            'body_size', 'upper_wick', 'lower_wick', 'candle_direction',
            'dist_ema_9', 'dist_ema_21', 'dist_ema_50',
            'adx', 'plus_di', 'minus_di'
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
    
    async def get_full_training_data(self, symbol: str) -> Dict:
        """Get ALL available historical data for comprehensive training"""
        logger.info(f"Starting full historical data fetch for {symbol}")
        
        # Fetch maximum historical data for different timeframes
        data_5m = await self.fetch_all_historical_data(symbol, '5m', max_candles=5000)
        data_1h = await self.fetch_all_historical_data(symbol, '1h', max_candles=5000)
        data_1d = await self.fetch_all_historical_data(symbol, '1d', max_candles=2000)
        
        if data_5m.empty:
            return {"error": "Could not fetch historical data"}
        
        # Calculate indicators
        data_5m = self.calculate_technical_indicators(data_5m)
        data_1h = self.calculate_technical_indicators(data_1h)
        data_1d = self.calculate_technical_indicators(data_1d)
        
        # Mathematical analysis
        math_analysis = {
            "support_resistance": self.math_model.find_support_resistance(data_1h),
            "fibonacci": self.math_model.fibonacci_retracement(data_1d),
            "trend": self.math_model.detect_trend(data_1d),
            "patterns": self.math_model.detect_patterns(data_5m),
            "volatility": self.math_model.calculate_volatility_regime(data_5m)
        }
        
        # Prepare features
        features = self.prepare_features(data_5m)
        features = self.scale_features(features, f'{symbol}_full', fit=True)
        
        # Create labels (next candle direction)
        data_5m['future_return'] = data_5m['close'].shift(-1) / data_5m['close'] - 1
        data_5m['label'] = (data_5m['future_return'] > 0).astype(int)
        labels = data_5m['label'].dropna().values
        
        # Align features and labels
        min_len = min(len(features) - 1, len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"Prepared {len(features)} samples for training")
        
        return {
            'features': features,
            'labels': labels,
            'math_analysis': math_analysis,
            'data_info': {
                'total_5m_candles': len(data_5m),
                'total_1h_candles': len(data_1h),
                'total_1d_candles': len(data_1d),
                'date_range': {
                    'start': str(data_5m.index[0]) if len(data_5m) > 0 else None,
                    'end': str(data_5m.index[-1]) if len(data_5m) > 0 else None
                }
            }
        }
    
    async def get_latest_data(self, symbol: str) -> Dict:
        """Get latest data for real-time prediction"""
        data_5m = await self.fetch_ohlcv(symbol, '5m', limit=100)
        data_1h = await self.fetch_ohlcv(symbol, '1h', limit=100)
        
        data_5m = self.calculate_technical_indicators(data_5m)
        data_1h = self.calculate_technical_indicators(data_1h)
        
        features = self.prepare_features(data_5m)
        
        if f'{symbol}_full' in self.scalers:
            features = self.scale_features(features, f'{symbol}_full', fit=False)
        
        current_price = float(data_5m['close'].iloc[-1]) if not data_5m.empty else 0
        current_atr = float(data_5m['atr'].iloc[-1]) if not data_5m.empty and 'atr' in data_5m.columns else 0
        
        # Get mathematical analysis
        math_analysis = {
            "support_resistance": self.math_model.find_support_resistance(data_1h),
            "trend": self.math_model.detect_trend(data_5m),
            "patterns": self.math_model.detect_patterns(data_5m),
            "volatility": self.math_model.calculate_volatility_regime(data_5m)
        }
        
        return {
            'features': features[-50:] if len(features) > 50 else features,
            'current_price': current_price,
            'current_atr': current_atr,
            'math_analysis': math_analysis,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def calculate_tp_sl(self, current_price: float, atr: float, direction: int, math_analysis: Dict = None) -> Dict:
        """Calculate TP/SL using ATR and mathematical levels"""
        # Base TP/SL from ATR
        if direction == 1:  # Long
            tp = current_price + (2.5 * atr)
            sl = current_price - (1.5 * atr)
        else:  # Short
            tp = current_price - (2.5 * atr)
            sl = current_price + (1.5 * atr)
        
        # Adjust based on support/resistance if available
        if math_analysis and 'support_resistance' in math_analysis:
            sr = math_analysis['support_resistance']
            
            if direction == 1 and sr.get('resistance'):
                # For long, use nearest resistance as TP
                resistances = [r for r in sr['resistance'] if r > current_price]
                if resistances:
                    tp = min(tp, min(resistances))
            
            if direction == 0 and sr.get('support'):
                # For short, use nearest support as TP
                supports = [s for s in sr['support'] if s < current_price]
                if supports:
                    tp = max(tp, max(supports))
        
        return {
            'take_profit': round(tp, 2),
            'stop_loss': round(sl, 2),
            'risk_reward': round(abs(tp - current_price) / abs(sl - current_price), 2) if sl != current_price else 1.67
        }


# Singleton instance
data_pipeline = EnhancedDataPipeline()
