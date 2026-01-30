"""
Advanced Data Pipeline with Multi-Timeframe, Feature Engineering, and Late Fusion Support
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler
import ta
import logging
import asyncio

logger = logging.getLogger(__name__)

class AdvancedDataPipeline:
    """
    Comprehensive data pipeline with:
    - Multi-timeframe data (5m, 15m, 1h, 4h, 1d)
    - Extended historical data (1-2 years)
    - Market regime detection
    - Cross-asset correlation
    - GARCH volatility features
    - Mock order book data
    - Mock on-chain metrics
    """
    
    TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
    TIMEFRAME_LIMITS = {
        '5m': 2000,    # ~7 days
        '15m': 2000,   # ~21 days
        '1h': 2000,    # ~83 days
        '4h': 2000,    # ~333 days
        '1d': 730,     # ~2 years
    }
    
    def __init__(self):
        self.exchange = ccxt.binanceus({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.scalers = {}
        self.data_cache = {}
        self.regime_model = None
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, 
                         since: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch OHLCV data with optional start date"""
        try:
            since_ms = int(since.timestamp() * 1000) if since else None
            
            # Fetch data in chunks for large requests
            all_data = []
            current_since = since_ms
            remaining = limit
            
            while remaining > 0:
                fetch_limit = min(remaining, 1000)
                data = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol, timeframe, current_since, fetch_limit
                )
                
                if not data:
                    break
                    
                all_data.extend(data)
                remaining -= len(data)
                
                if len(data) < fetch_limit:
                    break
                    
                current_since = data[-1][0] + 1
                await asyncio.sleep(0.1)  # Rate limiting
            
            if not all_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    async def fetch_multi_timeframe_data(self, symbol: str, 
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all timeframes"""
        data = {}
        
        for tf in self.TIMEFRAMES:
            limit = self.TIMEFRAME_LIMITS[tf]
            df = await self.fetch_ohlcv(symbol, tf, limit, start_date)
            
            if not df.empty and end_date:
                df = df[df.index <= end_date]
            
            data[tf] = df
            logger.info(f"Fetched {len(df)} rows for {symbol} {tf}")
        
        return data
    
    async def fetch_cross_asset_data(self, primary_symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch cross-asset data for correlation features"""
        cross_assets = {
            'ETH/USDT': None,
            'BTC/USDT': None,
        }
        
        for asset in cross_assets.keys():
            if asset != primary_symbol:
                df = await self.fetch_ohlcv(asset, '1h', 1000)
                cross_assets[asset] = df
        
        return cross_assets
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Trend Indicators
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_100'] = ta.sma(df['close'], length=100)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi_6'] = ta.rsi(df['close'], length=6)
        df['rsi_24'] = ta.rsi(df['close'], length=24)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands['BBU_20_2.0']
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR & Volatility
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
        
        # CCI
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        # MFI
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # VWAP (approximation)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Price ratios
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        df['price_sma200_ratio'] = df['close'] / df['sma_200']
        
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility_10'] = df['return_1'].rolling(10).std()
        df['volatility_20'] = df['return_1'].rolling(20).std()
        df['volatility_50'] = df['return_1'].rolling(50).std()
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        return df
    
    def calculate_garch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH-based volatility features"""
        if df.empty or len(df) < 100:
            return df
        
        df = df.copy()
        
        try:
            from arch import arch_model
            
            returns = df['return_1'].dropna() * 100  # Scale for GARCH
            
            if len(returns) > 100:
                # Fit GARCH(1,1) model
                model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                result = model.fit(disp='off', show_warning=False)
                
                # Get conditional volatility
                cond_vol = result.conditional_volatility
                df.loc[cond_vol.index, 'garch_volatility'] = cond_vol
                
                # Volatility forecast (1-step ahead)
                forecast = result.forecast(horizon=1)
                df['garch_forecast'] = np.nan
                df.iloc[-1, df.columns.get_loc('garch_forecast')] = np.sqrt(forecast.variance.values[-1, 0])
                
                # Fill forward GARCH volatility
                df['garch_volatility'] = df['garch_volatility'].ffill()
                
        except Exception as e:
            logger.warning(f"GARCH calculation failed: {e}")
            df['garch_volatility'] = df['volatility_20']
            df['garch_forecast'] = df['volatility_20']
        
        return df
    
    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime: Bull, Bear, or Sideways"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Simple regime detection based on moving averages and ADX
        df['regime'] = 'sideways'
        
        # Bull: Price above SMA50, SMA50 above SMA200, positive momentum
        bull_condition = (
            (df['close'] > df['sma_50']) & 
            (df['sma_50'] > df['sma_200']) &
            (df['return_20'] > 0)
        )
        
        # Bear: Price below SMA50, SMA50 below SMA200, negative momentum
        bear_condition = (
            (df['close'] < df['sma_50']) & 
            (df['sma_50'] < df['sma_200']) &
            (df['return_20'] < 0)
        )
        
        df.loc[bull_condition, 'regime'] = 'bull'
        df.loc[bear_condition, 'regime'] = 'bear'
        
        # Encode regime
        regime_map = {'bear': -1, 'sideways': 0, 'bull': 1}
        df['regime_encoded'] = df['regime'].map(regime_map)
        
        # Regime strength (based on ADX)
        df['regime_strength'] = df['adx'] / 100  # Normalize to 0-1
        
        return df
    
    def calculate_cross_asset_features(self, df: pd.DataFrame, 
                                       cross_data: Dict[str, pd.DataFrame],
                                       primary_symbol: str) -> pd.DataFrame:
        """Calculate cross-asset correlation features"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # BTC dominance proxy (if primary is not BTC)
        if primary_symbol != 'BTC/USDT' and 'BTC/USDT' in cross_data:
            btc_df = cross_data['BTC/USDT']
            if btc_df is not None and not btc_df.empty:
                # Align indices
                btc_returns = btc_df['close'].pct_change().reindex(df.index, method='ffill')
                df['btc_return'] = btc_returns
                df['btc_correlation'] = df['return_1'].rolling(20).corr(df['btc_return'])
        
        # ETH/BTC ratio (if primary is BTC)
        if primary_symbol == 'BTC/USDT' and 'ETH/USDT' in cross_data:
            eth_df = cross_data['ETH/USDT']
            if eth_df is not None and not eth_df.empty:
                eth_close = eth_df['close'].reindex(df.index, method='ffill')
                df['eth_btc_ratio'] = eth_close / df['close']
                df['eth_btc_ratio_change'] = df['eth_btc_ratio'].pct_change(5)
        
        return df
    
    def generate_mock_order_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mock order book features (replace with real API later)"""
        if df.empty:
            return df
        
        df = df.copy()
        np.random.seed(42)
        
        # Mock bid-ask spread (typically 0.01% - 0.1% for major pairs)
        base_spread = 0.0005  # 0.05%
        df['bid_ask_spread'] = base_spread * (1 + 0.5 * np.random.randn(len(df)))
        df['bid_ask_spread'] = df['bid_ask_spread'].clip(0.0001, 0.005)
        
        # Mock order flow imbalance (-1 to 1, where positive = more buying pressure)
        # Correlated with price direction
        df['order_flow_imbalance'] = 0.3 * np.sign(df['return_1']) + 0.7 * np.random.randn(len(df)) * 0.3
        df['order_flow_imbalance'] = df['order_flow_imbalance'].clip(-1, 1)
        
        # Mock bid/ask depth ratio
        df['depth_ratio'] = 1 + 0.2 * df['order_flow_imbalance'] + 0.1 * np.random.randn(len(df))
        df['depth_ratio'] = df['depth_ratio'].clip(0.5, 2.0)
        
        # Mock large order detection
        df['large_buy_orders'] = (np.random.rand(len(df)) > 0.9).astype(int)
        df['large_sell_orders'] = (np.random.rand(len(df)) > 0.9).astype(int)
        
        logger.info("Generated mock order book features (replace with real API)")
        
        return df
    
    def generate_mock_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mock on-chain metrics (replace with Glassnode/CryptoQuant API later)"""
        if df.empty:
            return df
        
        df = df.copy()
        np.random.seed(43)
        
        # Mock active addresses (normalized)
        base_activity = 0.5
        df['active_addresses'] = base_activity + 0.2 * np.sin(np.arange(len(df)) * 0.1) + 0.1 * np.random.randn(len(df))
        df['active_addresses'] = df['active_addresses'].clip(0.1, 1.0)
        
        # Mock exchange inflow/outflow ratio
        # < 1 = more outflow (bullish), > 1 = more inflow (bearish)
        df['exchange_flow_ratio'] = 1 + 0.1 * np.random.randn(len(df))
        df['exchange_flow_ratio'] = df['exchange_flow_ratio'].clip(0.7, 1.3)
        
        # Mock NUPL (Net Unrealized Profit/Loss) - ranges from -1 to 1
        df['nupl'] = 0.3 + 0.2 * np.cumsum(df['return_1'].fillna(0)) + 0.1 * np.random.randn(len(df))
        df['nupl'] = df['nupl'].clip(-0.5, 0.8)
        
        # Mock SOPR (Spent Output Profit Ratio) - around 1
        df['sopr'] = 1 + 0.05 * np.random.randn(len(df))
        df['sopr'] = df['sopr'].clip(0.9, 1.1)
        
        # Mock whale transactions
        df['whale_transactions'] = np.random.poisson(3, len(df))
        
        logger.info("Generated mock on-chain features (replace with real API)")
        
        return df
    
    def create_labels(self, df: pd.DataFrame, lookahead: int = 1, 
                     threshold: float = 0.0) -> pd.DataFrame:
        """Create binary labels for classification"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Future return
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Binary label: 1 = up, 0 = down
        df['label'] = (df['future_return'] > threshold).astype(int)
        
        return df
    
    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for training"""
        feature_columns = [
            # Price ratios
            'price_sma20_ratio', 'price_sma50_ratio', 'price_sma200_ratio',
            # Momentum
            'rsi', 'rsi_6', 'rsi_24', 'macd', 'macd_signal', 'macd_hist',
            'momentum_10', 'momentum_20', 'roc_10',
            # Trend
            'adx', 'di_plus', 'di_minus', 'cci',
            # Volatility
            'atr_percent', 'bb_width', 'bb_percent',
            'volatility_10', 'volatility_20', 'volatility_50',
            # Volume
            'volume_ratio', 'mfi',
            # Stochastic
            'stoch_k', 'stoch_d',
            # Returns
            'return_1', 'return_5', 'return_10', 'return_20',
            # GARCH
            'garch_volatility',
            # Regime
            'regime_encoded', 'regime_strength',
            # Order book (mock)
            'bid_ask_spread', 'order_flow_imbalance', 'depth_ratio',
            # On-chain (mock)
            'active_addresses', 'exchange_flow_ratio', 'nupl', 'sopr',
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        features = df[available_features].values
        
        return features, available_features
    
    def scale_features(self, features: np.ndarray, scaler_key: str, 
                      fit: bool = True) -> np.ndarray:
        """Scale features using RobustScaler"""
        if fit:
            self.scalers[scaler_key] = RobustScaler()
            scaled = self.scalers[scaler_key].fit_transform(features)
        else:
            if scaler_key in self.scalers:
                scaled = self.scalers[scaler_key].transform(features)
            else:
                scaled = features
        
        # Handle NaN/Inf
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return scaled
    
    async def prepare_training_data(self, symbol: str, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   timeframe: str = '1h',
                                   multi_timeframe: bool = False) -> Dict:
        """
        Prepare comprehensive training data with all features
        """
        logger.info(f"Preparing training data for {symbol}, timeframe={timeframe}, multi_tf={multi_timeframe}")
        
        if multi_timeframe:
            # Fetch all timeframes
            all_data = await self.fetch_multi_timeframe_data(symbol, start_date, end_date)
            
            # Process each timeframe
            processed_data = {}
            for tf, df in all_data.items():
                if not df.empty:
                    df = self.calculate_technical_indicators(df)
                    df = self.calculate_garch_features(df)
                    df = self.detect_market_regime(df)
                    df = self.generate_mock_order_book_features(df)
                    df = self.generate_mock_onchain_features(df)
                    df = self.create_labels(df)
                    processed_data[tf] = df
            
            # Use primary timeframe for training
            primary_df = processed_data.get(timeframe, pd.DataFrame())
            
        else:
            # Single timeframe
            limit = self.TIMEFRAME_LIMITS.get(timeframe, 1000)
            df = await self.fetch_ohlcv(symbol, timeframe, limit, start_date)
            
            if not df.empty and end_date:
                df = df[df.index <= end_date]
            
            # Apply all feature engineering
            df = self.calculate_technical_indicators(df)
            df = self.calculate_garch_features(df)
            df = self.detect_market_regime(df)
            
            # Cross-asset features
            cross_data = await self.fetch_cross_asset_data(symbol)
            df = self.calculate_cross_asset_features(df, cross_data, symbol)
            
            df = self.generate_mock_order_book_features(df)
            df = self.generate_mock_onchain_features(df)
            df = self.create_labels(df)
            
            primary_df = df
            processed_data = {timeframe: df}
        
        if primary_df.empty:
            return {
                'features': np.array([]),
                'labels': np.array([]),
                'feature_names': [],
                'data_info': {'error': 'No data available'}
            }
        
        # Remove rows with NaN labels
        primary_df = primary_df.dropna(subset=['label'])
        
        # Prepare features
        features, feature_names = self.prepare_features_for_training(primary_df)
        
        # Handle NaN in features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        labels = primary_df['label'].values
        
        # Scale features
        features = self.scale_features(features, f'{symbol}_{timeframe}', fit=True)
        
        # Data info
        data_info = {
            'symbol': symbol,
            'timeframe': timeframe,
            'samples': len(features),
            'features': len(feature_names),
            'class_distribution': {
                'up': int(np.sum(labels == 1)),
                'down': int(np.sum(labels == 0))
            },
            'date_range': {
                'start': str(primary_df.index.min()),
                'end': str(primary_df.index.max())
            },
            'multi_timeframe': multi_timeframe
        }
        
        logger.info(f"Prepared {len(features)} samples with {len(feature_names)} features")
        
        return {
            'features': features,
            'labels': labels,
            'feature_names': feature_names,
            'data_info': data_info,
            'processed_data': processed_data
        }
    
    async def get_latest_data(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Get latest data for prediction"""
        df = await self.fetch_ohlcv(symbol, timeframe, limit=200)
        
        if df.empty:
            return {
                'features': np.array([]),
                'current_price': 0.0,
                'current_atr': 0.0,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': 'No data available'
            }
        
        # Apply all feature engineering
        df = self.calculate_technical_indicators(df)
        df = self.calculate_garch_features(df)
        df = self.detect_market_regime(df)
        
        cross_data = await self.fetch_cross_asset_data(symbol)
        df = self.calculate_cross_asset_features(df, cross_data, symbol)
        
        df = self.generate_mock_order_book_features(df)
        df = self.generate_mock_onchain_features(df)
        
        features, feature_names = self.prepare_features_for_training(df)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler_key = f'{symbol}_{timeframe}'
        if scaler_key in self.scalers:
            features = self.scale_features(features, scaler_key, fit=False)
        
        current_price = float(df['close'].iloc[-1])
        current_atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.02
        
        # Handle NaN
        if np.isnan(current_price) or np.isinf(current_price):
            current_price = 0.0
        if np.isnan(current_atr) or np.isinf(current_atr):
            current_atr = current_price * 0.02
        
        return {
            'features': features[-100:] if len(features) > 100 else features,
            'current_price': current_price,
            'current_atr': current_atr,
            'regime': df['regime'].iloc[-1] if 'regime' in df.columns else 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def calculate_tp_sl(self, current_price: float, atr: float, 
                       direction: int, regime: str = 'sideways') -> Dict:
        """Calculate TP/SL with regime-adjusted multipliers"""
        if atr == 0 or np.isnan(atr) or np.isinf(atr):
            atr = current_price * 0.02
        
        if current_price == 0 or np.isnan(current_price) or np.isinf(current_price):
            return {'take_profit': 0.0, 'stop_loss': 0.0, 'risk_reward': 1.67}
        
        # Regime-adjusted multipliers
        regime_multipliers = {
            'bull': {'tp': 3.0, 'sl': 1.2},
            'bear': {'tp': 2.0, 'sl': 1.8},
            'sideways': {'tp': 2.5, 'sl': 1.5}
        }
        
        mult = regime_multipliers.get(regime, regime_multipliers['sideways'])
        
        if direction == 1:  # Long
            tp = current_price + (mult['tp'] * atr)
            sl = current_price - (mult['sl'] * atr)
        else:  # Short
            tp = current_price - (mult['tp'] * atr)
            sl = current_price + (mult['sl'] * atr)
        
        return {
            'take_profit': round(float(tp), 2),
            'stop_loss': round(float(sl), 2),
            'risk_reward': round(mult['tp'] / mult['sl'], 2)
        }


# Singleton instance
advanced_data_pipeline = AdvancedDataPipeline()
