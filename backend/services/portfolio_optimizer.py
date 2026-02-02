"""
Portfolio Optimizer
Implements multiple optimization strategies for crypto portfolio allocation
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

from services.correlation_analyzer import correlation_analyzer, DEFAULT_ASSETS
from services.multi_asset_predictor import multi_asset_predictor

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    max_weight_per_asset: float = 0.30  # Maximum 30% in any single asset
    min_weight_per_asset: float = 0.0   # Minimum weight (0 = can exclude)
    min_assets: int = 5                  # Minimum number of assets to include
    max_assets: int = 20                 # Maximum number of assets
    target_return: Optional[float] = None  # Target annual return (for min-risk optimization)


class PortfolioOptimizer:
    """
    Multi-strategy portfolio optimizer supporting:
    - Traditional Mean-Variance (Markowitz)
    - ML-enhanced optimization
    - Risk-based optimization
    """
    
    STRATEGIES = ['traditional_ml', 'deep_learning', 'rl_agent', 'hybrid']
    OBJECTIVES = ['max_sharpe', 'max_return', 'min_risk', 'risk_parity']
    HORIZONS = ['24h', '7d', '30d']
    
    def __init__(self):
        self.last_optimization: Optional[Dict] = None
        self.optimization_history: List[Dict] = []
        self.is_optimizing = False
    
    def _portfolio_return(self, weights: np.ndarray, expected_returns: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return np.dot(weights, expected_returns)
    
    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _portfolio_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate portfolio Sharpe ratio"""
        port_return = self._portfolio_return(weights, expected_returns)
        port_vol = self._portfolio_volatility(weights, cov_matrix)
        
        if port_vol == 0:
            return 0
        
        return (port_return - risk_free_rate) / port_vol
    
    def _neg_sharpe(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """Negative Sharpe ratio for minimization"""
        return -self._portfolio_sharpe(weights, expected_returns, cov_matrix, risk_free_rate)
    
    async def optimize_traditional_ml(
        self,
        assets: List[str],
        investment_amount: float,
        objective: str = 'max_sharpe',
        constraints: Optional[PortfolioConstraints] = None,
        horizon: str = '7d',
        use_ml_predictions: bool = True
    ) -> Dict:
        """
        Traditional Mean-Variance Optimization with ML-predicted returns
        
        Args:
            assets: List of assets to include
            investment_amount: Total amount to invest
            objective: Optimization objective
            constraints: Portfolio constraints
            horizon: Prediction horizon
            use_ml_predictions: Whether to use ML predictions for expected returns
        
        Returns:
            Optimized portfolio allocation
        """
        constraints = constraints or PortfolioConstraints()
        n_assets = len(assets)
        
        # Get correlation/covariance data
        if correlation_analyzer.covariance_matrix is None:
            correlation_analyzer.calculate_covariance_matrix()
        
        # Filter to available assets
        available_assets = [a for a in assets if a in correlation_analyzer.price_data]
        
        if len(available_assets) < constraints.min_assets:
            return {
                'status': 'error',
                'message': f'Not enough assets with data. Need {constraints.min_assets}, have {len(available_assets)}'
            }
        
        n_assets = len(available_assets)
        
        # Get expected returns
        if use_ml_predictions and multi_asset_predictor.is_trained:
            # Use ML predictions
            predictions = await multi_asset_predictor.predict_returns(available_assets, horizon)
            expected_returns = np.array([
                predictions.get(a, {}).get('expected_return', 0) / 100  # Convert from percentage
                for a in available_assets
            ])
        else:
            # Use historical mean returns
            hist_returns = correlation_analyzer.get_expected_returns()
            expected_returns = np.array([
                hist_returns.get(a, 0) for a in available_assets
            ])
        
        # Get covariance matrix for available assets
        cov_matrix = correlation_analyzer.covariance_matrix
        cov_subset = cov_matrix.loc[available_assets, available_assets].values
        
        # Handle any NaN in covariance matrix
        cov_subset = np.nan_to_num(cov_subset, nan=0.01)
        
        # Make covariance matrix positive semi-definite
        min_eig = np.min(np.linalg.eigvalsh(cov_subset))
        if min_eig < 0:
            cov_subset -= 1.1 * min_eig * np.eye(n_assets)
        
        # Initial guess: equal weights
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints
        bounds = tuple((constraints.min_weight_per_asset, constraints.max_weight_per_asset) 
                      for _ in range(n_assets))
        
        # Sum of weights = 1
        constraint_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Minimum number of assets constraint (approximated)
        # This is handled post-optimization by filtering small weights
        
        scipy_constraints = [constraint_sum]
        
        # Target return constraint for min-risk optimization
        if objective == 'min_risk' and constraints.target_return is not None:
            constraint_return = {
                'type': 'eq',
                'fun': lambda w: self._portfolio_return(w, expected_returns) - constraints.target_return
            }
            scipy_constraints.append(constraint_return)
        
        # Objective function based on strategy
        if objective == 'max_sharpe':
            objective_fun = lambda w: self._neg_sharpe(w, expected_returns, cov_subset)
        elif objective == 'max_return':
            objective_fun = lambda w: -self._portfolio_return(w, expected_returns)
        elif objective == 'min_risk':
            objective_fun = lambda w: self._portfolio_volatility(w, cov_subset)
        elif objective == 'risk_parity':
            # Risk parity: equal risk contribution from each asset
            def risk_parity_objective(w):
                port_vol = self._portfolio_volatility(w, cov_subset)
                if port_vol == 0:
                    return 1e10
                marginal_contrib = np.dot(cov_subset, w) / port_vol
                risk_contrib = w * marginal_contrib
                target_risk = port_vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)
            objective_fun = risk_parity_objective
        else:
            objective_fun = lambda w: self._neg_sharpe(w, expected_returns, cov_subset)
        
        # Optimize
        try:
            result = minimize(
                objective_fun,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
            
            optimal_weights = result.x
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fall back to equal weights
            optimal_weights = init_weights
        
        # Enforce minimum assets constraint by zeroing out smallest weights
        if constraints.min_assets > 0:
            non_zero_count = np.sum(optimal_weights > 0.01)
            if non_zero_count < constraints.min_assets:
                # Distribute to top assets by expected return
                sorted_idx = np.argsort(expected_returns)[::-1]
                optimal_weights = np.zeros(n_assets)
                for i in range(constraints.min_assets):
                    optimal_weights[sorted_idx[i]] = 1.0 / constraints.min_assets
        
        # Normalize weights to sum to 1
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        # Calculate portfolio metrics
        port_return = self._portfolio_return(optimal_weights, expected_returns)
        port_vol = self._portfolio_volatility(optimal_weights, cov_subset)
        port_sharpe = self._portfolio_sharpe(optimal_weights, expected_returns, cov_subset)
        
        # Build allocation result
        allocations = []
        for i, asset in enumerate(available_assets):
            weight = float(optimal_weights[i])
            if weight > 0.001:  # Filter out tiny allocations
                amount = investment_amount * weight
                allocations.append({
                    'symbol': asset,
                    'weight': round(weight * 100, 2),  # As percentage
                    'amount': round(amount, 2),
                    'expected_return': round(float(expected_returns[i]) * 100, 2)
                })
        
        # Sort by weight descending
        allocations.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'status': 'success',
            'strategy': 'traditional_ml',
            'objective': objective,
            'horizon': horizon,
            'investment_amount': investment_amount,
            'allocations': allocations,
            'metrics': {
                'expected_return': round(port_return * 100, 2),
                'volatility': round(port_vol * 100, 2),
                'sharpe_ratio': round(port_sharpe, 3),
                'num_assets': len(allocations)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def optimize_portfolio(
        self,
        assets: List[str],
        investment_amount: float,
        strategy: str = 'traditional_ml',
        objective: str = 'max_sharpe',
        constraints: Optional[PortfolioConstraints] = None,
        horizon: str = '7d',
        compare_all: bool = False
    ) -> Dict:
        """
        Main optimization entry point
        
        Args:
            assets: List of assets to include
            investment_amount: Total amount to invest
            strategy: Optimization strategy
            objective: Optimization objective
            constraints: Portfolio constraints
            horizon: Prediction horizon
            compare_all: Whether to run all strategies and compare
        
        Returns:
            Optimized portfolio allocation(s)
        """
        self.is_optimizing = True
        constraints = constraints or PortfolioConstraints()
        
        try:
            if compare_all:
                # Run all strategies and compare
                results = {}
                
                # Strategy A: Traditional + ML
                results['traditional_ml'] = await self.optimize_traditional_ml(
                    assets, investment_amount, objective, constraints, horizon, use_ml_predictions=True
                )
                
                # Strategy B: Deep Learning (placeholder - will be implemented in Phase 2)
                results['deep_learning'] = {
                    'status': 'pending',
                    'message': 'Deep Learning strategy will be implemented in Phase 2',
                    'strategy': 'deep_learning'
                }
                
                # Strategy C: RL Agent (placeholder - will be implemented in Phase 2)
                results['rl_agent'] = {
                    'status': 'pending',
                    'message': 'RL Agent strategy will be implemented in Phase 2',
                    'strategy': 'rl_agent'
                }
                
                # Strategy D: Hybrid (average of available strategies)
                if results['traditional_ml'].get('status') == 'success':
                    results['hybrid'] = results['traditional_ml'].copy()
                    results['hybrid']['strategy'] = 'hybrid'
                    results['hybrid']['note'] = 'Currently using Traditional+ML (other strategies pending)'
                else:
                    results['hybrid'] = {
                        'status': 'pending',
                        'message': 'Hybrid strategy requires at least one working strategy',
                        'strategy': 'hybrid'
                    }
                
                self.last_optimization = {
                    'type': 'comparison',
                    'results': results,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                return {
                    'status': 'success',
                    'type': 'comparison',
                    'strategies': results,
                    'recommended': 'traditional_ml',  # Best available strategy
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            else:
                # Run single strategy
                if strategy == 'traditional_ml':
                    result = await self.optimize_traditional_ml(
                        assets, investment_amount, objective, constraints, horizon
                    )
                elif strategy in ['deep_learning', 'rl_agent']:
                    result = {
                        'status': 'pending',
                        'message': f'{strategy} strategy will be implemented in Phase 2',
                        'strategy': strategy
                    }
                elif strategy == 'hybrid':
                    # For now, hybrid uses traditional_ml
                    result = await self.optimize_traditional_ml(
                        assets, investment_amount, objective, constraints, horizon
                    )
                    result['strategy'] = 'hybrid'
                else:
                    result = {
                        'status': 'error',
                        'message': f'Unknown strategy: {strategy}'
                    }
                
                self.last_optimization = result
                return result
                
        finally:
            self.is_optimizing = False
    
    def get_efficient_frontier(
        self,
        assets: List[str],
        n_points: int = 50
    ) -> Dict:
        """
        Calculate the efficient frontier for visualization
        
        Args:
            assets: List of assets
            n_points: Number of points on the frontier
        
        Returns:
            Efficient frontier data for charting
        """
        # Get available assets with data
        available_assets = [a for a in assets if a in correlation_analyzer.price_data]
        
        if len(available_assets) < 2:
            return {'status': 'error', 'message': 'Need at least 2 assets'}
        
        # Get expected returns and covariance
        expected_returns = correlation_analyzer.get_expected_returns()
        cov_matrix = correlation_analyzer.covariance_matrix
        
        exp_ret_arr = np.array([expected_returns.get(a, 0) for a in available_assets])
        cov_subset = cov_matrix.loc[available_assets, available_assets].values
        
        # Handle NaN
        exp_ret_arr = np.nan_to_num(exp_ret_arr, nan=0.0)
        cov_subset = np.nan_to_num(cov_subset, nan=0.01)
        
        n_assets = len(available_assets)
        
        # Calculate frontier points
        frontier_points = []
        
        # Target returns range from min to max expected return
        min_ret = np.min(exp_ret_arr)
        max_ret = np.max(exp_ret_arr)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        for target_ret in target_returns:
            # Minimize volatility for target return
            try:
                init_weights = np.array([1.0 / n_assets] * n_assets)
                bounds = tuple((0.0, 1.0) for _ in range(n_assets))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'eq', 'fun': lambda w: self._portfolio_return(w, exp_ret_arr) - target_ret}
                ]
                
                result = minimize(
                    lambda w: self._portfolio_volatility(w, cov_subset),
                    init_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500}
                )
                
                if result.success:
                    weights = result.x
                    vol = self._portfolio_volatility(weights, cov_subset)
                    ret = self._portfolio_return(weights, exp_ret_arr)
                    sharpe = (ret - 0.05) / vol if vol > 0 else 0
                    
                    frontier_points.append({
                        'return': round(ret * 100, 2),
                        'volatility': round(vol * 100, 2),
                        'sharpe': round(sharpe, 3)
                    })
                    
            except Exception:
                continue
        
        # Sort by volatility
        frontier_points.sort(key=lambda x: x['volatility'])
        
        return {
            'status': 'success',
            'frontier': frontier_points,
            'assets': available_assets
        }
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        return {
            'is_optimizing': self.is_optimizing,
            'last_optimization': self.last_optimization
        }


# Singleton instance
portfolio_optimizer = PortfolioOptimizer()
