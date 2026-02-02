"""
Multi-Asset Correlation Analyzer
Fetches data for multiple crypto assets and calculates correlation matrices
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio

from services.advanced_data_pipeline import advanced_data_pipeline

logger = logging.getLogger(__name__)

# Default supported assets (Top 20 by liquidity)
DEFAULT_ASSETS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "NEAR/USDT"
]


class CorrelationAnalyzer:
    """
    Analyzes correlations between multiple crypto assets for portfolio optimization
    """
    
    def __init__(self):
        self.assets = DEFAULT_ASSETS.copy()
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.returns_data: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
    
    def set_assets(self, assets: List[str]):
        """Set the list of assets to analyze"""
        self.assets = assets
        self.price_data = {}
        self.returns_data = {}
        self.correlation_matrix = None
        self.covariance_matrix = None
        logger.info(f"Assets set to: {assets}")
    
    def get_available_assets(self) -> List[str]:
        """Get list of available assets"""
        return DEFAULT_ASSETS.copy()
    
    async def fetch_multi_asset_data(
        self,
        assets: Optional[List[str]] = None,
        timeframe: str = '1d',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_periods: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple assets
        
        Args:
            assets: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframe: Candle timeframe
            start_date: Start date for data
            end_date: End date for data
            min_periods: Minimum number of periods required
        
        Returns:
            Dictionary mapping asset symbols to DataFrames
        """
        assets = assets or self.assets
        
        if not start_date:
            # Default to 1 year of data
            start_date = datetime.now(timezone.utc) - timedelta(days=365)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        logger.info(f"Fetching data for {len(assets)} assets from {start_date} to {end_date}")
        
        self.price_data = {}
        failed_assets = []
        
        for asset in assets:
            try:
                df = await advanced_data_pipeline.fetch_ohlcv(
                    symbol=asset,
                    timeframe=timeframe,
                    limit=1000000,
                    since=start_date,
                    until=end_date
                )
                
                if not df.empty and len(df) >= min_periods:
                    self.price_data[asset] = df
                    logger.info(f"Fetched {len(df)} candles for {asset}")
                else:
                    logger.warning(f"Insufficient data for {asset}: {len(df)} candles")
                    failed_assets.append(asset)
                    
            except Exception as e:
                logger.error(f"Failed to fetch {asset}: {e}")
                failed_assets.append(asset)
            
            # Rate limiting
            await asyncio.sleep(0.15)
        
        if failed_assets:
            logger.warning(f"Failed to fetch {len(failed_assets)} assets: {failed_assets}")
        
        self.last_update = datetime.now(timezone.utc)
        return self.price_data
    
    def calculate_returns(self, price_col: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate returns for all assets
        
        Args:
            price_col: Column to use for price (default: 'close')
        
        Returns:
            Dictionary mapping asset symbols to return Series
        """
        self.returns_data = {}
        
        for asset, df in self.price_data.items():
            if price_col in df.columns:
                # Calculate log returns for better statistical properties
                returns = np.log(df[price_col] / df[price_col].shift(1))
                self.returns_data[asset] = returns.dropna()
        
        return self.returns_data
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between all assets
        
        Returns:
            Correlation matrix as DataFrame
        """
        if not self.returns_data:
            self.calculate_returns()
        
        # Align all return series to common dates
        returns_df = pd.DataFrame(self.returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            logger.warning("No common dates found for correlation calculation")
            return pd.DataFrame()
        
        self.correlation_matrix = returns_df.corr()
        logger.info(f"Calculated correlation matrix for {len(self.correlation_matrix)} assets")
        
        return self.correlation_matrix
    
    def calculate_covariance_matrix(self, annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix between all assets
        
        Args:
            annualize: Whether to annualize the covariance (assumes daily data)
        
        Returns:
            Covariance matrix as DataFrame
        """
        if not self.returns_data:
            self.calculate_returns()
        
        # Align all return series to common dates
        returns_df = pd.DataFrame(self.returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return pd.DataFrame()
        
        cov_matrix = returns_df.cov()
        
        if annualize:
            # Annualize assuming 365 trading days for crypto
            cov_matrix = cov_matrix * 365
        
        self.covariance_matrix = cov_matrix
        return self.covariance_matrix
    
    def get_expected_returns(self, method: str = 'mean', annualize: bool = True) -> pd.Series:
        """
        Calculate expected returns for each asset
        
        Args:
            method: 'mean' for historical mean, 'ewm' for exponentially weighted
            annualize: Whether to annualize returns
        
        Returns:
            Series of expected returns per asset
        """
        if not self.returns_data:
            self.calculate_returns()
        
        returns_df = pd.DataFrame(self.returns_data)
        returns_df = returns_df.dropna()
        
        if method == 'mean':
            expected_returns = returns_df.mean()
        elif method == 'ewm':
            # Exponentially weighted mean (more weight on recent data)
            expected_returns = returns_df.ewm(span=30).mean().iloc[-1]
        else:
            expected_returns = returns_df.mean()
        
        if annualize:
            # Annualize assuming 365 trading days
            expected_returns = expected_returns * 365
        
        return expected_returns
    
    def get_volatility(self, annualize: bool = True) -> pd.Series:
        """
        Calculate volatility (standard deviation) for each asset
        
        Args:
            annualize: Whether to annualize volatility
        
        Returns:
            Series of volatilities per asset
        """
        if not self.returns_data:
            self.calculate_returns()
        
        returns_df = pd.DataFrame(self.returns_data)
        returns_df = returns_df.dropna()
        
        volatility = returns_df.std()
        
        if annualize:
            # Annualize assuming 365 trading days
            volatility = volatility * np.sqrt(365)
        
        return volatility
    
    def get_asset_statistics(self) -> Dict[str, Dict]:
        """
        Get comprehensive statistics for all assets
        
        Returns:
            Dictionary with statistics per asset
        """
        expected_returns = self.get_expected_returns()
        volatility = self.get_volatility()
        
        stats = {}
        for asset in self.price_data.keys():
            if asset in expected_returns.index and asset in volatility.index:
                ret = expected_returns[asset]
                vol = volatility[asset]
                sharpe = ret / vol if vol > 0 else 0
                
                # Get price data
                df = self.price_data[asset]
                current_price = float(df['close'].iloc[-1]) if not df.empty else 0
                
                stats[asset] = {
                    'symbol': asset,
                    'current_price': round(current_price, 4),
                    'expected_return': round(float(ret) * 100, 2),  # As percentage
                    'volatility': round(float(vol) * 100, 2),  # As percentage
                    'sharpe_ratio': round(float(sharpe), 3),
                    'data_points': len(df)
                }
        
        return stats
    
    def get_correlation_heatmap_data(self) -> Dict:
        """
        Get correlation matrix formatted for frontend heatmap visualization
        
        Returns:
            Dictionary with correlation data for charting
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {'assets': [], 'matrix': []}
        
        assets = list(self.correlation_matrix.columns)
        matrix = self.correlation_matrix.values.tolist()
        
        # Round values
        matrix = [[round(val, 3) for val in row] for row in matrix]
        
        return {
            'assets': [a.replace('/USDT', '') for a in assets],
            'matrix': matrix
        }
    
    def find_diversification_pairs(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Find asset pairs with low correlation (good for diversification)
        
        Args:
            threshold: Maximum correlation to consider as "diversified"
        
        Returns:
            List of tuples (asset1, asset2, correlation)
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []
        
        pairs = []
        assets = list(self.correlation_matrix.columns)
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:  # Avoid duplicates
                    corr = self.correlation_matrix.loc[asset1, asset2]
                    if abs(corr) < threshold:
                        pairs.append((asset1, asset2, round(float(corr), 3)))
        
        # Sort by absolute correlation (lowest first)
        pairs.sort(key=lambda x: abs(x[2]))
        
        return pairs


# Singleton instance
correlation_analyzer = CorrelationAnalyzer()
