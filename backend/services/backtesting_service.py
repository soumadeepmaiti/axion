"""
Backtesting Service for Crypto Trading System
Runs trained models against historical data to generate performance metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import threading
import logging
import asyncio
from dataclasses import dataclass, asdict

from services.data_pipeline import data_pipeline
from services.advanced_data_pipeline import advanced_data_pipeline
from services.advanced_training_service import advanced_training_service

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: str
    exit_time: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'signal', 'timeout'


@dataclass 
class BacktestResult:
    """Complete backtest results"""
    # Basic info
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    
    # Returns metrics
    total_return: float
    total_return_percent: float
    annualized_return: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float  # in hours
    
    # Additional metrics
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    
    # Time series data
    equity_curve: List[Dict]
    drawdown_curve: List[Dict]
    trades: List[Dict]
    monthly_returns: List[Dict]


class BacktestingService:
    """
    Service for backtesting trading strategies using trained models
    """
    
    def __init__(self):
        self.is_running = False
        self.backtest_thread = None
        self.stop_requested = False
        
        self.status = {
            'is_running': False,
            'progress': 0,
            'current_date': None,
            'total_trades': 0,
            'current_pnl': 0,
            'start_time': None,
            'error': None
        }
        
        self.result: Optional[BacktestResult] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
    
    def get_status(self) -> Dict:
        """Get current backtest status"""
        return self.status.copy()
    
    def get_result(self) -> Optional[Dict]:
        """Get backtest result"""
        if self.result:
            return asdict(self.result)
        return None
    
    def stop_backtest(self) -> Dict:
        """Stop running backtest"""
        if self.is_running:
            self.stop_requested = True
            return {"status": "stopping", "message": "Backtest stop requested"}
        return {"status": "not_running", "message": "No backtest in progress"}
    
    async def run_backtest(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,  # 10% of capital per trade
        use_stop_loss: bool = True,
        stop_loss_pct: float = 0.02,  # 2%
        use_take_profit: bool = True,
        take_profit_pct: float = 0.04,  # 4%
        max_hold_time: int = 24,  # hours
        min_confidence: float = 0.6,  # minimum prediction confidence to trade
        commission: float = 0.001,  # 0.1% per trade
    ) -> Dict:
        """
        Run backtest with specified parameters
        """
        if self.is_running:
            return {"error": "Backtest already in progress"}
        
        # Check if model is loaded
        if advanced_training_service.model is None and advanced_training_service.ensemble_model is None:
            return {"error": "No trained model loaded. Please train or load a model first."}
        
        self.is_running = True
        self.stop_requested = False
        self.trades = []
        self.equity_curve = []
        self.result = None
        
        self.status = {
            'is_running': True,
            'progress': 0,
            'current_date': None,
            'total_trades': 0,
            'current_pnl': 0,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'error': None
        }
        
        try:
            # Fetch historical data
            logger.info(f"Fetching historical data for backtest: {symbol} {timeframe}")
            
            df = await data_pipeline.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                max_candles=10000
            )
            
            if df.empty or len(df) < 100:
                self.is_running = False
                self.status['is_running'] = False
                self.status['error'] = "Insufficient data for backtest"
                return {"error": "Insufficient data for backtest (need at least 100 candles)"}
            
            # Prepare features using advanced data pipeline
            logger.info(f"Preparing features for {len(df)} candles")
            
            features_df = advanced_data_pipeline.prepare_features(df.reset_index())
            
            if features_df.empty:
                self.is_running = False
                self.status['is_running'] = False
                self.status['error'] = "Failed to prepare features"
                return {"error": "Failed to prepare features"}
            
            # Run the backtest simulation
            result = await self._simulate_trading(
                df=df,
                features_df=features_df,
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=initial_capital,
                position_size=position_size,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_take_profit=use_take_profit,
                take_profit_pct=take_profit_pct,
                max_hold_time=max_hold_time,
                min_confidence=min_confidence,
                commission=commission
            )
            
            self.result = result
            self.is_running = False
            self.status['is_running'] = False
            self.status['progress'] = 100
            
            return {"status": "completed", "result": asdict(result)}
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            self.is_running = False
            self.status['is_running'] = False
            self.status['error'] = str(e)
            return {"error": str(e)}
    
    async def _simulate_trading(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        initial_capital: float,
        position_size: float,
        use_stop_loss: bool,
        stop_loss_pct: float,
        use_take_profit: bool,
        take_profit_pct: float,
        max_hold_time: int,
        min_confidence: float,
        commission: float
    ) -> BacktestResult:
        """
        Simulate trading based on model predictions
        """
        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0
        max_drawdown_pct = 0
        
        trades: List[Trade] = []
        equity_curve: List[Dict] = []
        daily_returns: List[float] = []
        
        # Position tracking
        position = None  # {'direction': 'long'/'short', 'entry_price': float, 'entry_time': str, 'size': float}
        
        # Prepare sequences for prediction
        sequence_length = advanced_training_service.status.get('config', {}).get('sequence_length', 50)
        if sequence_length is None:
            sequence_length = 50
        
        # Scale features
        feature_columns = [col for col in features_df.columns if col != 'timestamp']
        features_array = features_df[feature_columns].values
        
        if advanced_training_service.scaler is not None:
            features_scaled = advanced_training_service.scaler.transform(features_array)
        else:
            # Simple normalization
            features_scaled = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-8)
        
        total_bars = len(df) - sequence_length
        
        # Get timeframe in hours for duration calculations
        tf_hours = self._timeframe_to_hours(timeframe)
        
        logger.info(f"Starting backtest simulation with {total_bars} bars")
        
        for i in range(sequence_length, len(df)):
            if self.stop_requested:
                logger.info("Backtest stopped by user")
                break
            
            current_idx = i
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Update progress
            progress = int(((i - sequence_length) / total_bars) * 100)
            self.status['progress'] = progress
            self.status['current_date'] = str(current_time)
            self.status['total_trades'] = len(trades)
            self.status['current_pnl'] = capital - initial_capital
            
            # Check existing position
            if position is not None:
                entry_price = position['entry_price']
                entry_time = position['entry_time']
                direction = position['direction']
                trade_size = position['size']
                
                # Calculate current PnL
                if direction == 'long':
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price
                
                # Check exit conditions
                exit_reason = None
                exit_price = current_price
                
                # Stop loss check (use low/high for more realistic fills)
                if use_stop_loss:
                    if direction == 'long' and current_low <= entry_price * (1 - stop_loss_pct):
                        exit_reason = 'stop_loss'
                        exit_price = entry_price * (1 - stop_loss_pct)
                    elif direction == 'short' and current_high >= entry_price * (1 + stop_loss_pct):
                        exit_reason = 'stop_loss'
                        exit_price = entry_price * (1 + stop_loss_pct)
                
                # Take profit check
                if exit_reason is None and use_take_profit:
                    if direction == 'long' and current_high >= entry_price * (1 + take_profit_pct):
                        exit_reason = 'take_profit'
                        exit_price = entry_price * (1 + take_profit_pct)
                    elif direction == 'short' and current_low <= entry_price * (1 - take_profit_pct):
                        exit_reason = 'take_profit'
                        exit_price = entry_price * (1 - take_profit_pct)
                
                # Time-based exit
                if exit_reason is None and max_hold_time > 0:
                    entry_dt = pd.to_datetime(entry_time)
                    hours_held = (current_time - entry_dt).total_seconds() / 3600
                    if hours_held >= max_hold_time:
                        exit_reason = 'timeout'
                        exit_price = current_price
                
                # Close position if exit triggered
                if exit_reason:
                    if direction == 'long':
                        pnl_pct = (exit_price - entry_price) / entry_price - commission * 2
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price - commission * 2
                    
                    pnl = trade_size * pnl_pct
                    capital += pnl
                    
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=str(current_time),
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=trade_size,
                        pnl=pnl,
                        pnl_percent=pnl_pct * 100,
                        exit_reason=exit_reason
                    )
                    trades.append(trade)
                    position = None
            
            # Make prediction for new position
            if position is None and i < len(df) - 1:  # Don't trade on last bar
                # Prepare sequence
                seq_start = i - sequence_length
                sequence = features_scaled[seq_start:i]
                
                if len(sequence) == sequence_length:
                    # Get prediction
                    prediction = self._get_model_prediction(sequence)
                    
                    if prediction is not None:
                        direction_pred = prediction['direction']
                        confidence = prediction['confidence']
                        
                        # Only trade if confidence exceeds threshold
                        if confidence >= min_confidence:
                            trade_capital = capital * position_size
                            
                            if direction_pred == 1:  # Long
                                position = {
                                    'direction': 'long',
                                    'entry_price': current_price,
                                    'entry_time': str(current_time),
                                    'size': trade_capital
                                }
                            elif direction_pred == 0:  # Short
                                position = {
                                    'direction': 'short',
                                    'entry_price': current_price,
                                    'entry_time': str(current_time),
                                    'size': trade_capital
                                }
            
            # Update equity curve
            equity = capital
            if position is not None:
                # Add unrealized PnL
                if position['direction'] == 'long':
                    unrealized = position['size'] * ((current_price - position['entry_price']) / position['entry_price'])
                else:
                    unrealized = position['size'] * ((position['entry_price'] - current_price) / position['entry_price'])
                equity += unrealized
            
            equity_curve.append({
                'timestamp': str(current_time),
                'equity': equity,
                'capital': capital,
                'position': position['direction'] if position else None
            })
            
            # Track drawdown
            if equity > peak_capital:
                peak_capital = equity
            drawdown = peak_capital - equity
            drawdown_pct = drawdown / peak_capital if peak_capital > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
            
            # Yield control periodically
            if i % 100 == 0:
                await asyncio.sleep(0)
        
        # Close any remaining position at end
        if position is not None:
            current_price = df['close'].iloc[-1]
            entry_price = position['entry_price']
            direction = position['direction']
            trade_size = position['size']
            
            if direction == 'long':
                pnl_pct = (current_price - entry_price) / entry_price - commission * 2
            else:
                pnl_pct = (entry_price - current_price) / entry_price - commission * 2
            
            pnl = trade_size * pnl_pct
            capital += pnl
            
            trade = Trade(
                entry_time=position['entry_time'],
                exit_time=str(df.index[-1]),
                direction=direction,
                entry_price=entry_price,
                exit_price=current_price,
                size=trade_size,
                pnl=pnl,
                pnl_percent=pnl_pct * 100,
                exit_reason='end_of_data'
            )
            trades.append(trade)
        
        # Calculate metrics
        result = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            symbol=symbol,
            timeframe=timeframe,
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            initial_capital=initial_capital,
            final_capital=capital,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct
        )
        
        self.trades = trades
        self.equity_curve = equity_curve
        
        return result
    
    def _get_model_prediction(self, sequence: np.ndarray) -> Optional[Dict]:
        """Get prediction from loaded model"""
        try:
            # Reshape for model input
            X = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if advanced_training_service.ensemble_model is not None:
                # Ensemble model
                pred = advanced_training_service.ensemble_model.predict(X)
                direction = int(pred[0])
                confidence = 0.7  # Ensemble doesn't provide probability
            elif advanced_training_service.model is not None:
                # Neural network model
                pred = advanced_training_service.model.predict(X, verbose=0)
                
                if isinstance(pred, list):
                    # Multi-output model
                    direction_pred = pred[0][0]
                else:
                    direction_pred = pred[0]
                
                if len(direction_pred) > 1:
                    # Softmax output
                    direction = int(np.argmax(direction_pred))
                    confidence = float(np.max(direction_pred))
                else:
                    # Sigmoid output
                    prob = float(direction_pred[0])
                    direction = 1 if prob > 0.5 else 0
                    confidence = prob if direction == 1 else 1 - prob
            else:
                return None
            
            return {
                'direction': direction,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[Dict],
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        final_capital: float,
        max_drawdown: float,
        max_drawdown_pct: float
    ) -> BacktestResult:
        """Calculate all performance metrics"""
        
        # Basic returns
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100 if initial_capital > 0 else 0
        
        # Calculate period in days
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            days = (end_dt - start_dt).days
            years = days / 365.25 if days > 0 else 1
        except:
            days = 365
            years = 1
        
        # Annualized return
        if years > 0 and initial_capital > 0:
            annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Trade duration
        durations = []
        for t in trades:
            try:
                entry = pd.to_datetime(t.entry_time)
                exit = pd.to_datetime(t.exit_time)
                duration_hours = (exit - entry).total_seconds() / 3600
                durations.append(duration_hours)
            except:
                pass
        avg_duration = np.mean(durations) if durations else 0
        
        # Best/worst trades
        pnls = [t.pnl for t in trades]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        for t in trades:
            if t.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        # Calculate daily returns for Sharpe/Sortino
        daily_returns = self._calculate_daily_returns(equity_curve)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(daily_returns) > 1:
            avg_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe_ratio = (avg_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        if len(daily_returns) > 1:
            negative_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = (np.mean(daily_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = (annualized_return / (max_drawdown_pct * 100)) if max_drawdown_pct > 0 else 0
        
        # Calculate drawdown curve
        drawdown_curve = self._calculate_drawdown_curve(equity_curve)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percent=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_pct * 100,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            equity_curve=equity_curve[-500:] if len(equity_curve) > 500 else equity_curve,  # Limit size
            drawdown_curve=drawdown_curve[-500:] if len(drawdown_curve) > 500 else drawdown_curve,
            trades=[asdict(t) for t in trades],
            monthly_returns=monthly_returns
        )
    
    def _calculate_daily_returns(self, equity_curve: List[Dict]) -> List[float]:
        """Calculate daily returns from equity curve"""
        if len(equity_curve) < 2:
            return []
        
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to daily
        daily = df['equity'].resample('D').last().dropna()
        returns = daily.pct_change().dropna().tolist()
        
        return returns
    
    def _calculate_drawdown_curve(self, equity_curve: List[Dict]) -> List[Dict]:
        """Calculate drawdown over time"""
        if not equity_curve:
            return []
        
        result = []
        peak = equity_curve[0]['equity']
        
        for point in equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0
            result.append({
                'timestamp': point['timestamp'],
                'drawdown': drawdown * 100  # as percentage
            })
        
        return result
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict]) -> List[Dict]:
        """Calculate monthly returns"""
        if len(equity_curve) < 2:
            return []
        
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to monthly
        monthly = df['equity'].resample('ME').last().dropna()
        returns = monthly.pct_change().dropna()
        
        result = []
        for date, ret in returns.items():
            result.append({
                'month': date.strftime('%Y-%m'),
                'return': ret * 100
            })
        
        return result
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Convert timeframe string to hours"""
        tf_map = {
            '1m': 1/60, '5m': 5/60, '15m': 0.25, '30m': 0.5,
            '1h': 1, '4h': 4, '1d': 24, '1w': 168
        }
        return tf_map.get(timeframe, 1)


# Global instance
backtesting_service = BacktestingService()
