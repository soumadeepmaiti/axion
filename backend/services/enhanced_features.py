"""
Enhanced Feature Engineering Module
Adds Market Microstructure, Cross-Market, Time-Based, and Advanced Technical Features
All features use mock data - replace with real APIs when available
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Bitcoin Halving Dates
HALVING_DATES = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),  # First halving
    datetime(2016, 7, 9, tzinfo=timezone.utc),    # Second halving
    datetime(2020, 5, 11, tzinfo=timezone.utc),   # Third halving
    datetime(2024, 4, 20, tzinfo=timezone.utc),   # Fourth halving (estimated)
]


class MarketMicrostructureFeatures:
    """
    Market Microstructure Features (MOCK - Replace with Binance Futures API)
    - Funding Rate
    - Open Interest
    - Liquidation Data
    - Exchange Inflow/Outflow
    """
    
    @staticmethod
    def generate_funding_rate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock funding rate data
        Real API: Binance GET /fapi/v1/fundingRate
        Funding rate is typically -0.01% to +0.1%, updated every 8 hours
        """
        df = df.copy()
        np.random.seed(44)
        
        n = len(df)
        # Base funding rate around 0.01% with some variation
        base_rate = 0.0001
        
        # Funding rate correlates with price momentum
        momentum = df['close'].pct_change(24).fillna(0)
        
        # Generate funding rate: positive when price rising (longs pay shorts)
        df['funding_rate'] = base_rate + 0.0005 * momentum + 0.0002 * np.random.randn(n)
        df['funding_rate'] = df['funding_rate'].clip(-0.001, 0.003)  # -0.1% to 0.3%
        
        # Cumulative funding (cost of holding position)
        df['cumulative_funding'] = df['funding_rate'].cumsum()
        
        # Funding rate momentum
        df['funding_rate_ma'] = df['funding_rate'].rolling(8).mean()
        df['funding_rate_std'] = df['funding_rate'].rolling(24).std()
        
        # Extreme funding indicator
        df['extreme_funding'] = (df['funding_rate'].abs() > 0.001).astype(int)
        
        logger.info("Generated mock funding rate features (replace with Binance Futures API)")
        return df
    
    @staticmethod
    def generate_open_interest(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock open interest data
        Real API: Binance GET /fapi/v1/openInterest
        """
        df = df.copy()
        np.random.seed(45)
        
        n = len(df)
        # Base OI that trends with price
        price_normalized = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
        
        base_oi = 50000 + 30000 * price_normalized  # OI in BTC terms
        noise = np.random.randn(n) * 2000
        
        df['open_interest'] = base_oi + noise
        df['open_interest'] = df['open_interest'].clip(20000, 100000)
        
        # OI changes
        df['oi_change'] = df['open_interest'].pct_change()
        df['oi_change_1h'] = df['open_interest'].pct_change(1)
        df['oi_change_4h'] = df['open_interest'].pct_change(4)
        df['oi_change_24h'] = df['open_interest'].pct_change(24)
        
        # OI momentum
        df['oi_ma_20'] = df['open_interest'].rolling(20).mean()
        df['oi_ratio'] = df['open_interest'] / df['oi_ma_20']
        
        # OI divergence from price (useful signal)
        price_change = df['close'].pct_change(24)
        oi_change = df['open_interest'].pct_change(24)
        df['oi_price_divergence'] = oi_change - price_change
        
        logger.info("Generated mock open interest features (replace with Binance Futures API)")
        return df
    
    @staticmethod
    def generate_liquidation_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock liquidation data
        Real API: Binance WebSocket or Coinglass API
        """
        df = df.copy()
        np.random.seed(46)
        
        n = len(df)
        
        # Liquidations correlate with volatility and price moves
        volatility = df['close'].pct_change().rolling(24).std().fillna(0.02)
        price_change = df['close'].pct_change().abs().fillna(0)
        
        # Long liquidations happen on drops
        df['long_liquidations'] = np.maximum(0, -df['close'].pct_change() * 1000000 * volatility + np.random.exponential(50000, n))
        df['long_liquidations'] = df['long_liquidations'].clip(0, 500000)
        
        # Short liquidations happen on pumps
        df['short_liquidations'] = np.maximum(0, df['close'].pct_change() * 1000000 * volatility + np.random.exponential(50000, n))
        df['short_liquidations'] = df['short_liquidations'].clip(0, 500000)
        
        # Total and net liquidations
        df['total_liquidations'] = df['long_liquidations'] + df['short_liquidations']
        df['net_liquidations'] = df['long_liquidations'] - df['short_liquidations']
        
        # Liquidation ratio
        df['liquidation_ratio'] = df['long_liquidations'] / (df['short_liquidations'] + 1)
        
        # Cumulative liquidations (24h)
        df['liquidations_24h'] = df['total_liquidations'].rolling(24).sum()
        
        logger.info("Generated mock liquidation features (replace with Coinglass API)")
        return df
    
    @staticmethod
    def generate_exchange_flows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock exchange inflow/outflow data
        Real API: Glassnode, CryptoQuant
        """
        df = df.copy()
        np.random.seed(47)
        
        n = len(df)
        
        # Base flow that varies with price action
        price_momentum = df['close'].pct_change(24).fillna(0)
        
        # Inflow: tends to increase during rallies (profit taking)
        df['exchange_inflow'] = 5000 + 2000 * price_momentum + np.random.exponential(1000, n)
        df['exchange_inflow'] = df['exchange_inflow'].clip(1000, 20000)
        
        # Outflow: tends to increase during dips (accumulation)
        df['exchange_outflow'] = 5000 - 1500 * price_momentum + np.random.exponential(1000, n)
        df['exchange_outflow'] = df['exchange_outflow'].clip(1000, 20000)
        
        # Net flow (negative = bullish, coins leaving exchanges)
        df['exchange_netflow'] = df['exchange_inflow'] - df['exchange_outflow']
        
        # Flow ratio
        df['exchange_flow_ratio'] = df['exchange_inflow'] / (df['exchange_outflow'] + 1)
        
        # Rolling metrics
        df['netflow_ma_7d'] = df['exchange_netflow'].rolling(168).mean()  # 7 days in hours
        df['netflow_cumulative'] = df['exchange_netflow'].cumsum()
        
        logger.info("Generated mock exchange flow features (replace with Glassnode/CryptoQuant API)")
        return df


class CrossMarketFeatures:
    """
    Cross-Market Signal Features (MOCK - Replace with real market data APIs)
    - S&P 500 correlation
    - DXY (Dollar Index)
    - Gold correlation
    - Fear & Greed Index
    """
    
    @staticmethod
    def generate_sp500_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock S&P 500 correlation features
        Real API: Yahoo Finance, Alpha Vantage
        """
        df = df.copy()
        np.random.seed(48)
        
        n = len(df)
        
        # Simulate S&P 500 price movement (correlated with BTC during risk-on)
        btc_returns = df['close'].pct_change().fillna(0)
        correlation = 0.4  # BTC-SPX correlation
        
        sp500_returns = correlation * btc_returns + np.sqrt(1 - correlation**2) * np.random.randn(n) * 0.01
        df['sp500_return'] = sp500_returns
        
        # Cumulative S&P 500 proxy
        df['sp500_price'] = 4500 * (1 + sp500_returns).cumprod()
        
        # Rolling correlation
        df['btc_sp500_corr_30d'] = df['close'].pct_change().rolling(720).corr(df['sp500_return'])
        
        # S&P 500 momentum
        df['sp500_momentum'] = df['sp500_price'].pct_change(24)
        df['sp500_ma_50'] = df['sp500_price'].rolling(50).mean()
        df['sp500_above_ma'] = (df['sp500_price'] > df['sp500_ma_50']).astype(int)
        
        logger.info("Generated mock S&P 500 features (replace with Yahoo Finance API)")
        return df
    
    @staticmethod
    def generate_dxy_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock DXY (Dollar Index) features
        Real API: Yahoo Finance (DX-Y.NYB), TradingView
        BTC typically moves inverse to DXY
        """
        df = df.copy()
        np.random.seed(49)
        
        n = len(df)
        
        # DXY moves inversely to BTC (negative correlation)
        btc_returns = df['close'].pct_change().fillna(0)
        correlation = -0.3  # Negative correlation
        
        dxy_returns = correlation * btc_returns + np.sqrt(1 - correlation**2) * np.random.randn(n) * 0.005
        
        # DXY typically ranges 90-110
        df['dxy_price'] = 103 * (1 + dxy_returns).cumprod()
        df['dxy_price'] = df['dxy_price'].clip(90, 115)
        
        df['dxy_return'] = dxy_returns
        df['dxy_momentum'] = df['dxy_price'].pct_change(24)
        
        # DXY strength indicator
        df['dxy_ma_20'] = df['dxy_price'].rolling(20).mean()
        df['dxy_strength'] = (df['dxy_price'] - df['dxy_ma_20']) / df['dxy_ma_20']
        
        # BTC-DXY correlation
        df['btc_dxy_corr'] = df['close'].pct_change().rolling(168).corr(df['dxy_return'])
        
        logger.info("Generated mock DXY features (replace with Yahoo Finance API)")
        return df
    
    @staticmethod
    def generate_gold_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock Gold (XAU) features
        Real API: Yahoo Finance (GC=F), Alpha Vantage
        BTC sometimes acts as "digital gold"
        """
        df = df.copy()
        np.random.seed(50)
        
        n = len(df)
        
        # Gold has slight positive correlation with BTC
        btc_returns = df['close'].pct_change().fillna(0)
        correlation = 0.2
        
        gold_returns = correlation * btc_returns + np.sqrt(1 - correlation**2) * np.random.randn(n) * 0.008
        
        # Gold price around $2000
        df['gold_price'] = 2000 * (1 + gold_returns).cumprod()
        df['gold_price'] = df['gold_price'].clip(1800, 2400)
        
        df['gold_return'] = gold_returns
        df['gold_momentum'] = df['gold_price'].pct_change(24)
        
        # BTC/Gold ratio (digital gold narrative)
        df['btc_gold_ratio'] = df['close'] / df['gold_price']
        df['btc_gold_ratio_ma'] = df['btc_gold_ratio'].rolling(50).mean()
        
        logger.info("Generated mock Gold features (replace with Yahoo Finance API)")
        return df
    
    @staticmethod
    def generate_fear_greed_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock Fear & Greed Index
        Real API: alternative.me/crypto/fear-and-greed-index/
        Range: 0 (Extreme Fear) to 100 (Extreme Greed)
        """
        df = df.copy()
        np.random.seed(51)
        
        n = len(df)
        
        # Fear & Greed correlates with price momentum and volatility
        momentum = df['close'].pct_change(24).fillna(0)
        volatility = df['close'].pct_change().rolling(24).std().fillna(0.02)
        
        # Base index
        base_fg = 50 + 200 * momentum - 500 * volatility + np.random.randn(n) * 10
        df['fear_greed_index'] = base_fg.clip(0, 100)
        
        # Smooth it (index is daily)
        df['fear_greed_index'] = df['fear_greed_index'].rolling(24).mean()
        
        # Classification
        df['fg_extreme_fear'] = (df['fear_greed_index'] < 20).astype(int)
        df['fg_fear'] = ((df['fear_greed_index'] >= 20) & (df['fear_greed_index'] < 40)).astype(int)
        df['fg_neutral'] = ((df['fear_greed_index'] >= 40) & (df['fear_greed_index'] < 60)).astype(int)
        df['fg_greed'] = ((df['fear_greed_index'] >= 60) & (df['fear_greed_index'] < 80)).astype(int)
        df['fg_extreme_greed'] = (df['fear_greed_index'] >= 80).astype(int)
        
        # Contrarian signal (extreme fear = buy, extreme greed = sell)
        df['fg_contrarian'] = (50 - df['fear_greed_index']) / 50
        
        logger.info("Generated mock Fear & Greed Index (replace with alternative.me API)")
        return df


class TimeBasedFeatures:
    """
    Time-Based Features for Seasonality
    - Trading sessions (Asian/EU/US)
    - Day of week
    - Month seasonality
    - Days since halving
    """
    
    @staticmethod
    def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate all time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Trading Sessions (UTC times)
        # Asian: 00:00-08:00 UTC
        # European: 08:00-16:00 UTC
        # US: 14:00-22:00 UTC (overlaps with EU)
        df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['session_european'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['session_us'] = ((df['hour'] >= 14) & (df['hour'] < 22)).astype(int)
        df['session_overlap'] = ((df['hour'] >= 14) & (df['hour'] < 16)).astype(int)  # EU-US overlap
        
        # Weekend effect
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Monday effect (historically volatile)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        
        # Month-end effect
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # Cyclical encoding for hour (sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        logger.info("Generated time-based features")
        return df
    
    @staticmethod
    def generate_halving_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bitcoin halving cycle features
        Halving occurs every ~4 years, historically bullish
        """
        df = df.copy()
        
        def days_since_last_halving(timestamp):
            for i in range(len(HALVING_DATES) - 1, -1, -1):
                if timestamp >= HALVING_DATES[i]:
                    return (timestamp - HALVING_DATES[i]).days
            return 0
        
        def days_to_next_halving(timestamp):
            for halving in HALVING_DATES:
                if timestamp < halving:
                    return (halving - timestamp).days
            # Estimate next halving (~4 years after last)
            return (HALVING_DATES[-1] + pd.Timedelta(days=1460) - timestamp).days
        
        # Calculate halving metrics
        df['days_since_halving'] = df.index.map(lambda x: days_since_last_halving(x))
        df['days_to_next_halving'] = df.index.map(lambda x: max(0, days_to_next_halving(x)))
        
        # Halving cycle position (0 = just happened, 1 = about to happen)
        cycle_length = 1460  # ~4 years in days
        df['halving_cycle_position'] = (df['days_since_halving'] % cycle_length) / cycle_length
        
        # Cyclical encoding
        df['halving_cycle_sin'] = np.sin(2 * np.pi * df['halving_cycle_position'])
        df['halving_cycle_cos'] = np.cos(2 * np.pi * df['halving_cycle_position'])
        
        # Pre-halving indicator (6 months before)
        df['pre_halving'] = (df['days_to_next_halving'] < 180).astype(int)
        
        # Post-halving indicator (6 months after)
        df['post_halving'] = (df['days_since_halving'] < 180).astype(int)
        
        logger.info("Generated halving cycle features")
        return df


class AdvancedTechnicalFeatures:
    """
    Advanced Technical Analysis Features
    - Wyckoff Analysis
    - Elliott Wave (simplified)
    - Ichimoku Cloud
    - Volume Profile
    """
    
    @staticmethod
    def generate_wyckoff_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Wyckoff accumulation/distribution features
        Based on price-volume relationship
        """
        df = df.copy()
        
        # Volume-Price Analysis
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        price_change = df['close'].pct_change()
        
        # Wyckoff Effort vs Result
        # High volume + small price move = effort without result (potential reversal)
        df['effort_vs_result'] = df['volume_ratio'] / (price_change.abs() + 0.001)
        df['effort_vs_result'] = df['effort_vs_result'].clip(0, 100)
        
        # Accumulation indicator: price down on low volume, up on high volume
        df['wyckoff_accum'] = 0.0
        up_days = price_change > 0
        down_days = price_change < 0
        high_vol = df['volume_ratio'] > 1.2
        low_vol = df['volume_ratio'] < 0.8
        
        # Accumulation signals
        df.loc[up_days & high_vol, 'wyckoff_accum'] = 1
        df.loc[down_days & low_vol, 'wyckoff_accum'] = 1
        
        # Distribution signals
        df.loc[up_days & low_vol, 'wyckoff_accum'] = -1
        df.loc[down_days & high_vol, 'wyckoff_accum'] = -1
        
        # Rolling Wyckoff score
        df['wyckoff_score'] = df['wyckoff_accum'].rolling(20).sum()
        
        # Spring/Upthrust detection (simplified)
        df['spring'] = 0
        df['upthrust'] = 0
        
        low_20 = df['low'].rolling(20).min()
        high_20 = df['high'].rolling(20).max()
        
        # Spring: price dips below support but closes above
        df.loc[(df['low'] < low_20.shift(1)) & (df['close'] > low_20.shift(1)), 'spring'] = 1
        
        # Upthrust: price breaks above resistance but closes below
        df.loc[(df['high'] > high_20.shift(1)) & (df['close'] < high_20.shift(1)), 'upthrust'] = 1
        
        logger.info("Generated Wyckoff features")
        return df
    
    @staticmethod
    def generate_elliott_wave_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate simplified Elliott Wave features
        Based on swing high/low detection
        """
        df = df.copy()
        
        # Detect swing highs and lows
        window = 5
        
        df['swing_high'] = 0
        df['swing_low'] = 0
        
        for i in range(window, len(df) - window):
            # Swing high: highest point in window
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                df.iloc[i, df.columns.get_loc('swing_high')] = 1
            # Swing low: lowest point in window
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                df.iloc[i, df.columns.get_loc('swing_low')] = 1
        
        # Count waves (simplified)
        df['wave_count_up'] = df['swing_high'].rolling(50).sum()
        df['wave_count_down'] = df['swing_low'].rolling(50).sum()
        
        # Wave momentum
        df['wave_momentum'] = df['wave_count_up'] - df['wave_count_down']
        
        # Fibonacci retracement levels (from recent swing)
        df['swing_high_price'] = df['high'].rolling(50).max()
        df['swing_low_price'] = df['low'].rolling(50).min()
        df['swing_range'] = df['swing_high_price'] - df['swing_low_price']
        
        df['fib_236'] = df['swing_high_price'] - 0.236 * df['swing_range']
        df['fib_382'] = df['swing_high_price'] - 0.382 * df['swing_range']
        df['fib_500'] = df['swing_high_price'] - 0.500 * df['swing_range']
        df['fib_618'] = df['swing_high_price'] - 0.618 * df['swing_range']
        
        # Price position relative to Fibonacci levels
        df['price_vs_fib_382'] = (df['close'] - df['fib_382']) / df['swing_range']
        df['price_vs_fib_618'] = (df['close'] - df['fib_618']) / df['swing_range']
        
        logger.info("Generated Elliott Wave features")
        return df
    
    @staticmethod
    def generate_ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Ichimoku Cloud indicators
        """
        df = df.copy()
        
        # Tenkan-sen (Conversion Line): 9-period high+low / 2
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line): 26-period high+low / 2
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period high+low / 2, plotted 26 periods ahead
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close plotted 26 periods behind
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        # Cloud color (bullish if Senkou A > Senkou B)
        df['ichimoku_cloud_bullish'] = (df['ichimoku_senkou_a'] > df['ichimoku_senkou_b']).astype(int)
        
        # Price vs Cloud
        cloud_top = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        cloud_bottom = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
        
        df['price_above_cloud'] = (df['close'] > cloud_top).astype(int)
        df['price_below_cloud'] = (df['close'] < cloud_bottom).astype(int)
        df['price_in_cloud'] = ((df['close'] >= cloud_bottom) & (df['close'] <= cloud_top)).astype(int)
        
        # TK Cross (Tenkan crosses Kijun)
        df['tk_cross_bullish'] = ((df['ichimoku_tenkan'] > df['ichimoku_kijun']) & 
                                  (df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1))).astype(int)
        df['tk_cross_bearish'] = ((df['ichimoku_tenkan'] < df['ichimoku_kijun']) & 
                                  (df['ichimoku_tenkan'].shift(1) >= df['ichimoku_kijun'].shift(1))).astype(int)
        
        # Cloud thickness (volatility indicator)
        df['cloud_thickness'] = (df['ichimoku_senkou_a'] - df['ichimoku_senkou_b']).abs() / df['close']
        
        logger.info("Generated Ichimoku Cloud features")
        return df
    
    @staticmethod
    def generate_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Volume Profile features
        POC (Point of Control), VAH (Value Area High), VAL (Value Area Low)
        """
        df = df.copy()
        
        # Rolling Volume Profile (simplified using volume-weighted price levels)
        window = 100
        
        # Calculate VWAP as proxy for POC
        df['vwap_100'] = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
        
        # Value Area (where 70% of volume traded)
        # Simplified: use price standard deviation
        df['price_std_100'] = df['close'].rolling(window).std()
        
        # POC (Point of Control) - price level with most volume
        # Approximated by VWAP
        df['volume_poc'] = df['vwap_100']
        
        # VAH (Value Area High) - POC + 1 std dev
        df['volume_vah'] = df['volume_poc'] + df['price_std_100']
        
        # VAL (Value Area Low) - POC - 1 std dev
        df['volume_val'] = df['volume_poc'] - df['price_std_100']
        
        # Price position relative to Volume Profile
        df['price_vs_poc'] = (df['close'] - df['volume_poc']) / df['price_std_100']
        
        df['price_above_vah'] = (df['close'] > df['volume_vah']).astype(int)
        df['price_below_val'] = (df['close'] < df['volume_val']).astype(int)
        df['price_in_value_area'] = ((df['close'] >= df['volume_val']) & 
                                      (df['close'] <= df['volume_vah'])).astype(int)
        
        # Volume at price levels
        df['volume_concentration'] = df['volume'].rolling(window).sum() / df['volume'].rolling(window * 5).sum()
        
        logger.info("Generated Volume Profile features")
        return df


class WhaleTrackingFeatures:
    """
    Whale Wallet Tracking Features (MOCK - Replace with on-chain APIs)
    """
    
    @staticmethod
    def generate_whale_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mock whale tracking features
        Real API: Whale Alert, Glassnode, Santiment
        """
        df = df.copy()
        np.random.seed(52)
        
        n = len(df)
        
        # Number of whale transactions (>$1M)
        df['whale_tx_count'] = np.random.poisson(5, n)
        
        # Whale buy/sell volume
        price_momentum = df['close'].pct_change(24).fillna(0)
        
        # Whales tend to buy during dips
        df['whale_buy_volume'] = np.maximum(0, 10000 - 50000 * price_momentum + np.random.exponential(5000, n))
        df['whale_sell_volume'] = np.maximum(0, 10000 + 50000 * price_momentum + np.random.exponential(5000, n))
        
        df['whale_net_flow'] = df['whale_buy_volume'] - df['whale_sell_volume']
        
        # Whale accumulation score
        df['whale_accumulation'] = df['whale_net_flow'].rolling(24).sum()
        df['whale_accumulation_ma'] = df['whale_accumulation'].rolling(168).mean()  # 7 day MA
        
        # Large transfer alerts
        df['whale_alert'] = (df['whale_tx_count'] > 10).astype(int)
        
        # Exchange whale deposits (bearish) vs withdrawals (bullish)
        df['whale_exchange_deposit'] = np.random.exponential(2000, n)
        df['whale_exchange_withdrawal'] = np.random.exponential(2500, n)  # Slightly more withdrawals on average
        df['whale_exchange_netflow'] = df['whale_exchange_deposit'] - df['whale_exchange_withdrawal']
        
        logger.info("Generated mock whale tracking features (replace with Whale Alert/Glassnode API)")
        return df


def apply_all_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all enhanced features to dataframe"""
    
    # Market Microstructure
    df = MarketMicrostructureFeatures.generate_funding_rate(df)
    df = MarketMicrostructureFeatures.generate_open_interest(df)
    df = MarketMicrostructureFeatures.generate_liquidation_data(df)
    df = MarketMicrostructureFeatures.generate_exchange_flows(df)
    
    # Cross-Market Signals
    df = CrossMarketFeatures.generate_sp500_features(df)
    df = CrossMarketFeatures.generate_dxy_features(df)
    df = CrossMarketFeatures.generate_gold_features(df)
    df = CrossMarketFeatures.generate_fear_greed_index(df)
    
    # Time-Based Features
    df = TimeBasedFeatures.generate_time_features(df)
    df = TimeBasedFeatures.generate_halving_features(df)
    
    # Advanced Technical
    df = AdvancedTechnicalFeatures.generate_wyckoff_features(df)
    df = AdvancedTechnicalFeatures.generate_elliott_wave_features(df)
    df = AdvancedTechnicalFeatures.generate_ichimoku_features(df)
    df = AdvancedTechnicalFeatures.generate_volume_profile_features(df)
    
    # Whale Tracking
    df = WhaleTrackingFeatures.generate_whale_features(df)
    
    logger.info(f"Applied all enhanced features. Total columns: {len(df.columns)}")
    
    return df
