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

from services.enhanced_features import apply_all_enhanced_features

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
        self.exchanges = {}
        self.active_exchange_name = 'binance'
        self._init_default_exchange()
        self.scalers = {}
        self.data_cache = {}
        self.regime_model = None
    
    def _init_default_exchange(self):
        """Initialize default exchange (Binance)"""
        self.exchanges['binance'] = ccxt.binanceus({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.exchange = self.exchanges['binance']
    
    def configure_exchange(self, exchange_name: str, api_key: str = None, 
                          api_secret: str = None, passphrase: str = None):
        """Configure an exchange with API credentials"""
        exchange_configs = {
            'binance': {
                'class': ccxt.binanceus,
                'options': {'defaultType': 'spot'}
            },
            'okx': {
                'class': ccxt.okx,
                'options': {'defaultType': 'spot'}
            },
            'kucoin': {
                'class': ccxt.kucoin,
                'options': {'defaultType': 'spot'}
            },
            'bybit': {
                'class': ccxt.bybit,
                'options': {'defaultType': 'spot'}
            },
            'kraken': {
                'class': ccxt.kraken,
                'options': {}
            },
            'coinbase': {
                'class': ccxt.coinbase,
                'options': {}
            }
        }
        
        if exchange_name not in exchange_configs:
            logger.warning(f"Unknown exchange: {exchange_name}")
            return False
        
        config = exchange_configs[exchange_name]
        
        try:
            exchange_params = {
                'enableRateLimit': True,
                'options': config['options']
            }
            
            if api_key and api_secret:
                exchange_params['apiKey'] = api_key
                exchange_params['secret'] = api_secret
                
                # OKX and KuCoin require passphrase
                if passphrase and exchange_name in ['okx', 'kucoin']:
                    exchange_params['password'] = passphrase
            
            self.exchanges[exchange_name] = config['class'](exchange_params)
            logger.info(f"Configured exchange: {exchange_name} (authenticated: {bool(api_key)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure {exchange_name}: {e}")
            return False
    
    def set_active_exchange(self, exchange_name: str):
        """Set the active exchange for data fetching"""
        if exchange_name in self.exchanges:
            self.active_exchange_name = exchange_name
            self.exchange = self.exchanges[exchange_name]
            logger.info(f"Active exchange set to: {exchange_name}")
            return True
        else:
            # Try to initialize without auth
            self.configure_exchange(exchange_name)
            if exchange_name in self.exchanges:
                self.active_exchange_name = exchange_name
                self.exchange = self.exchanges[exchange_name]
                return True
        return False
    
    def get_active_exchange(self) -> str:
        """Get the currently active exchange name"""
        return self.active_exchange_name
    
    async def test_exchange_connection(self, exchange_name: str = None) -> Dict:
        """Test connection to an exchange"""
        exchange_name = exchange_name or self.active_exchange_name
        
        if exchange_name not in self.exchanges:
            return {'success': False, 'error': f'Exchange {exchange_name} not configured'}
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Try to fetch ticker
            ticker = await asyncio.to_thread(exchange.fetch_ticker, 'BTC/USDT')
            
            # Check if authenticated
            is_authenticated = bool(exchange.apiKey and exchange.secret)
            
            # Try to fetch balance if authenticated
            balance = None
            if is_authenticated:
                try:
                    balance = await asyncio.to_thread(exchange.fetch_balance)
                    balance = {k: v for k, v in balance.get('total', {}).items() if v and v > 0}
                except Exception as e:
                    logger.warning(f"Could not fetch balance: {e}")
            
            return {
                'success': True,
                'exchange': exchange_name,
                'authenticated': is_authenticated,
                'ticker': {
                    'symbol': 'BTC/USDT',
                    'last': ticker.get('last'),
                    'bid': ticker.get('bid'),
                    'ask': ticker.get('ask')
                },
                'balance': balance
            }
            
        except Exception as e:
            logger.error(f"Exchange connection test failed: {e}")
            return {'success': False, 'exchange': exchange_name, 'error': str(e)}
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, 
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data with support for arbitrary date ranges.
        
        For large date ranges (e.g., 2015 to today), this method will automatically
        fetch data in chunks to avoid API timeouts and rate limits.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '1d')
            limit: Maximum number of candles (used if no date range specified)
            since: Start datetime for data fetching
            until: End datetime for data fetching (defaults to now)
        """
        try:
            # Convert datetime to milliseconds
            since_ms = int(since.timestamp() * 1000) if since else None
            until_ms = int(until.timestamp() * 1000) if until else int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Calculate timeframe in milliseconds for chunking
            timeframe_ms = self._get_timeframe_ms(timeframe)
            
            # If we have a date range, calculate how many candles we need
            if since_ms and until_ms:
                total_candles_needed = int((until_ms - since_ms) / timeframe_ms) + 1
                logger.info(f"Fetching {symbol} {timeframe} from {since} to {until} (~{total_candles_needed} candles)")
            else:
                total_candles_needed = limit
            
            # Fetch data in chunks (most exchanges limit to 1000-1500 candles per request)
            all_data = []
            current_since = since_ms
            max_candles_per_request = 1000
            total_fetched = 0
            
            while True:
                # Calculate how many candles to fetch in this request
                fetch_limit = min(max_candles_per_request, total_candles_needed - total_fetched) if since_ms else min(max_candles_per_request, limit - total_fetched)
                
                if fetch_limit <= 0:
                    break
                
                try:
                    data = await asyncio.to_thread(
                        self.exchange.fetch_ohlcv,
                        symbol, timeframe, current_since, fetch_limit
                    )
                except Exception as fetch_error:
                    logger.warning(f"Fetch error at {current_since}: {fetch_error}")
                    # If we have some data, return what we have
                    if all_data:
                        break
                    raise
                
                if not data:
                    logger.info(f"No more data available after {total_fetched} candles")
                    break
                
                # Filter out data beyond our end date
                if until_ms:
                    data = [d for d in data if d[0] <= until_ms]
                
                if not data:
                    break
                
                all_data.extend(data)
                total_fetched += len(data)
                
                # Log progress for large fetches
                if total_candles_needed > 1000 and total_fetched % 5000 == 0:
                    progress = (total_fetched / total_candles_needed) * 100 if total_candles_needed > 0 else 0
                    logger.info(f"Fetch progress: {total_fetched}/{total_candles_needed} candles ({progress:.1f}%)")
                
                # Check if we've reached the end date
                if until_ms and data[-1][0] >= until_ms:
                    logger.info(f"Reached end date after {total_fetched} candles")
                    break
                
                # Check if we got less than requested (end of available data)
                if len(data) < fetch_limit:
                    logger.info(f"Reached end of available data after {total_fetched} candles")
                    break
                
                # Check if we've hit our limit
                if not since_ms and total_fetched >= limit:
                    break
                
                # Move to next chunk - start from the last candle's timestamp + 1ms
                current_since = data[-1][0] + 1
                
                # Rate limiting - be nice to the exchange API
                await asyncio.sleep(0.1)
            
            if not all_data:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()  # Ensure chronological order
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h
    
    async def fetch_multi_timeframe_data(self, symbol: str, 
                                         start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all timeframes with full date range support"""
        data = {}
        
        for tf in self.TIMEFRAMES:
            # Use the new date-range based fetching
            if start_date and end_date:
                df = await self.fetch_ohlcv(symbol, tf, limit=100000, since=start_date, until=end_date)
            else:
                limit = self.TIMEFRAME_LIMITS[tf]
                df = await self.fetch_ohlcv(symbol, tf, limit, start_date)
            
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
        """Calculate comprehensive technical indicators using ta library"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Trend Indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_100'] = ta.trend.sma_indicator(df['close'], window=100)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_6'] = ta.momentum.rsi(df['close'], window=6)
        df['rsi_24'] = ta.momentum.rsi(df['close'], window=24)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        # ATR & Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # MFI
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        
        # OBV
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # VWAP (approximation)
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=14)
        
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
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        
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
    
    def prepare_features_for_training(self, df: pd.DataFrame, real_data_only: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix for training"""
        
        # REAL features - from actual OHLCV data + calculated technical indicators
        real_feature_columns = [
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
        ]
        
        # MOCK features - only include if not real_data_only
        mock_feature_columns = [
            # Order book (mock)
            'bid_ask_spread', 'order_flow_imbalance', 'depth_ratio',
            # On-chain (mock)
            'active_addresses', 'exchange_flow_ratio', 'nupl', 'sopr',
            # Market Microstructure (mock)
            'funding_rate', 'cumulative_funding', 'funding_rate_ma', 'extreme_funding',
            'open_interest', 'oi_change_24h', 'oi_ratio', 'oi_price_divergence',
            'liquidation_intensity', 'liquidation_imbalance', 'liquidation_ma',
            'premium_index', 'premium_ma', 'premium_reversal', 'extreme_premium',
            # Cross-Market (mock)
            'dxy_correlation', 'gold_correlation', 'spx_correlation',
            'risk_on_score', 'macro_regime', 'rate_sensitivity',
            'funding_spread', 'exchange_premium', 'arbitrage_opportunity', 'cross_exchange_flow',
            'sentiment_score', 'sentiment_ma', 'sentiment_momentum',
            'fear_greed', 'social_volume', 'social_sentiment',
            'news_sentiment', 'news_volume', 'sentiment_divergence',
            # On-Chain Advanced (mock)
            'mvrv_ratio', 'nvt_signal', 'realized_cap_change',
            'stablecoin_supply_ratio', 'stablecoin_inflow', 'buying_power_index',
            'whale_accumulation', 'exchange_whale_ratio', 'smart_money_flow',
            'network_value_signal', 'velocity', 'dormancy_flow',
            'puell_multiple', 'reserve_risk', 'hodl_waves_signal',
        ]
        
        if real_data_only:
            feature_columns = real_feature_columns
            logger.info(f"Using {len(feature_columns)} REAL features only (no mocked data)")
        else:
            feature_columns = real_feature_columns + mock_feature_columns
            logger.info(f"Using {len(feature_columns)} features (including mocked data)")
        
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
                                   multi_timeframe: bool = False,
                                   real_data_only: bool = True) -> Dict:
        """
        Prepare comprehensive training data with all features
        
        Args:
            real_data_only: If True, only use real OHLCV + technical indicators (no mocked data)
        """
        logger.info(f"Preparing training data for {symbol}, timeframe={timeframe}, real_data_only={real_data_only}")
        
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
                    if not real_data_only:
                        df = self.generate_mock_order_book_features(df)
                        df = self.generate_mock_onchain_features(df)
                    df = self.create_labels(df)
                    processed_data[tf] = df
            
            # Use primary timeframe for training
            primary_df = processed_data.get(timeframe, pd.DataFrame())
            
        else:
            # Single timeframe - calculate proper limit based on date range
            default_limit = self.TIMEFRAME_LIMITS.get(timeframe, 1000)
            
            # Calculate limit based on date range if provided
            if start_date and end_date:
                # Calculate number of candles between start and end date
                timeframe_minutes = {
                    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                    '1h': 60, '2h': 120, '4h': 240,
                    '1d': 1440, '1w': 10080
                }
                minutes_per_candle = timeframe_minutes.get(timeframe, 60)
                
                # Parse dates if they're strings
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                
                time_diff = end_date - start_date
                calculated_limit = int(time_diff.total_seconds() / 60 / minutes_per_candle) + 1
                limit = min(calculated_limit, 5000)  # Cap at 5000 candles
                logger.info(f"Date range: {start_date} to {end_date}, calculated {calculated_limit} candles, using {limit}")
            elif start_date:
                # If only start date, fetch from start to now
                limit = default_limit
                logger.info(f"Fetching {limit} candles from {start_date}")
            else:
                limit = default_limit
                logger.info(f"No date range specified, fetching last {limit} candles")
            
            df = await self.fetch_ohlcv(symbol, timeframe, limit, start_date)
            
            # Apply end date filter
            if not df.empty and end_date:
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                # Make end_date timezone aware if df.index is timezone aware
                if df.index.tz is not None and end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=df.index.tz)
                df = df[df.index <= end_date]
                logger.info(f"After end_date filter: {len(df)} candles")
            
            # Apply start date filter (in case fetch_ohlcv returned earlier data)
            if not df.empty and start_date:
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if df.index.tz is not None and start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=df.index.tz)
                df = df[df.index >= start_date]
                logger.info(f"After start_date filter: {len(df)} candles")
            
            # Apply all feature engineering
            df = self.calculate_technical_indicators(df)
            df = self.calculate_garch_features(df)
            df = self.detect_market_regime(df)
            
            # Cross-asset features (real data from exchanges)
            cross_data = await self.fetch_cross_asset_data(symbol)
            df = self.calculate_cross_asset_features(df, cross_data, symbol)
            
            if not real_data_only:
                # Only add mock features if explicitly requested
                df = self.generate_mock_order_book_features(df)
                df = self.generate_mock_onchain_features(df)
                df = apply_all_enhanced_features(df)
            else:
                logger.info("Using REAL DATA ONLY - skipping mocked on-chain, sentiment, order book features")
            
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
        
        # Prepare features - use real features only if specified
        features, feature_names = self.prepare_features_for_training(primary_df, real_data_only=real_data_only)
        
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
