"""
Multi-Asset Return Predictor
Trains ML models to predict returns for multiple crypto assets
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from sklearn.preprocessing import RobustScaler

from services.correlation_analyzer import correlation_analyzer, DEFAULT_ASSETS
from services.advanced_data_pipeline import advanced_data_pipeline

logger = logging.getLogger(__name__)


class MultiAssetPredictor:
    """
    Predicts future returns for multiple crypto assets using ML models
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}  # Asset -> trained model
        self.scalers: Dict[str, RobustScaler] = {}
        self.predictions: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_status = {
            'is_training': False,
            'current_asset': None,
            'progress': 0,
            'total_assets': 0,
            'completed_assets': []
        }
    
    async def prepare_features_for_asset(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and labels for a single asset
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Calculate technical indicators
        df = advanced_data_pipeline.calculate_technical_indicators(df.copy())
        
        # Create labels: future return (next period)
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)
        
        # Feature columns (subset of real features)
        feature_cols = [
            'price_sma20_ratio', 'price_sma50_ratio',
            'rsi', 'rsi_6', 'rsi_24',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'atr_percent', 'bb_width', 'bb_percent',
            'volatility_10', 'volatility_20',
            'volume_ratio', 'mfi',
            'stoch_k', 'stoch_d',
            'return_1', 'return_5', 'return_10', 'return_20',
            'momentum_10', 'roc_10'
        ]
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Drop rows with NaN in features or labels
        df_clean = df.dropna(subset=available_cols + ['label'])
        
        if df_clean.empty:
            return np.array([]), np.array([]), []
        
        features = df_clean[available_cols].values
        labels = df_clean['label'].values
        
        # Handle any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features, labels, available_cols
    
    async def train_single_asset_model(
        self,
        symbol: str,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train an LSTM model for a single asset
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Scale features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[symbol] = scaler
            
            # Create sequences
            seq_length = min(30, len(features_scaled) // 3)
            X, y = [], []
            
            for i in range(seq_length, len(features_scaled)):
                X.append(features_scaled[i-seq_length:i])
                y.append(labels[i])
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
                return {'status': 'failed', 'reason': 'insufficient_data'}
            
            # Train/val split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
                Dropout(0.2),
                BatchNormalization(),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            self.models[symbol] = {
                'model': model,
                'seq_length': seq_length,
                'accuracy': float(val_acc),
                'trained_at': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Trained model for {symbol}: accuracy={val_acc:.4f}")
            
            return {
                'status': 'success',
                'accuracy': float(val_acc),
                'epochs_run': len(history.history['loss'])
            }
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    async def train_all_assets(
        self,
        assets: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = '1d',
        epochs: int = 50
    ) -> Dict:
        """
        Train prediction models for all assets
        """
        assets = assets or DEFAULT_ASSETS
        
        self.training_status = {
            'is_training': True,
            'current_asset': None,
            'progress': 0,
            'total_assets': len(assets),
            'completed_assets': [],
            'start_time': datetime.now(timezone.utc).isoformat()
        }
        
        # Fetch data for all assets
        logger.info(f"Fetching data for {len(assets)} assets...")
        await correlation_analyzer.fetch_multi_asset_data(
            assets=assets,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        results = {}
        
        for i, asset in enumerate(assets):
            self.training_status['current_asset'] = asset
            self.training_status['progress'] = int((i / len(assets)) * 100)
            
            if asset not in correlation_analyzer.price_data:
                results[asset] = {'status': 'failed', 'reason': 'no_data'}
                continue
            
            df = correlation_analyzer.price_data[asset]
            features, labels, feature_names = await self.prepare_features_for_asset(asset, df)
            
            if len(features) == 0:
                results[asset] = {'status': 'failed', 'reason': 'no_features'}
                continue
            
            self.feature_names = feature_names
            
            result = await self.train_single_asset_model(
                symbol=asset,
                features=features,
                labels=labels,
                epochs=epochs
            )
            
            results[asset] = result
            self.training_status['completed_assets'].append(asset)
            
            # Small delay between training
            await asyncio.sleep(0.1)
        
        self.is_trained = True
        self.training_status['is_training'] = False
        self.training_status['progress'] = 100
        self.training_status['end_time'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate overall statistics
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        avg_accuracy = np.mean([r.get('accuracy', 0) for r in results.values() if r.get('status') == 'success'])
        
        return {
            'status': 'completed',
            'total_assets': len(assets),
            'successful': successful,
            'failed': len(assets) - successful,
            'average_accuracy': round(float(avg_accuracy) * 100, 2) if not np.isnan(avg_accuracy) else 0,
            'results': results
        }
    
    async def predict_returns(
        self,
        assets: Optional[List[str]] = None,
        horizon: str = '7d'
    ) -> Dict[str, Dict]:
        """
        Predict future returns for assets
        
        Args:
            assets: List of assets to predict
            horizon: Prediction horizon ('24h', '7d', '30d')
        
        Returns:
            Dictionary with predictions per asset
        """
        assets = assets or list(self.models.keys())
        
        if not assets:
            return {}
        
        # Horizon multiplier for return scaling
        horizon_multipliers = {
            '24h': 1,
            '7d': 7,
            '30d': 30
        }
        multiplier = horizon_multipliers.get(horizon, 7)
        
        predictions = {}
        
        for asset in assets:
            if asset not in self.models:
                continue
            
            model_info = self.models[asset]
            model = model_info['model']
            seq_length = model_info['seq_length']
            
            try:
                # Get latest data
                if asset not in correlation_analyzer.price_data:
                    continue
                
                df = correlation_analyzer.price_data[asset]
                df = advanced_data_pipeline.calculate_technical_indicators(df.copy())
                
                # Prepare features
                features, _, _ = await self.prepare_features_for_asset(asset, df)
                
                if len(features) < seq_length:
                    continue
                
                # Scale and reshape for prediction
                if asset in self.scalers:
                    features_scaled = self.scalers[asset].transform(features)
                else:
                    features_scaled = features
                
                # Get last sequence
                X = features_scaled[-seq_length:].reshape(1, seq_length, -1)
                
                # Predict
                prob = float(model.predict(X, verbose=0)[0][0])
                
                # Convert probability to expected return estimate
                # This is a simplified mapping
                direction = 1 if prob > 0.5 else -1
                confidence = abs(prob - 0.5) * 2
                
                # Estimate return based on historical volatility
                if asset in correlation_analyzer.returns_data:
                    daily_vol = float(correlation_analyzer.returns_data[asset].std())
                else:
                    daily_vol = 0.02  # Default 2% daily volatility
                
                # Expected return = direction * confidence * volatility * horizon
                expected_return = direction * confidence * daily_vol * np.sqrt(multiplier)
                
                predictions[asset] = {
                    'symbol': asset,
                    'probability_up': round(prob, 4),
                    'direction': 'UP' if prob > 0.5 else 'DOWN',
                    'confidence': round(confidence * 100, 1),
                    'expected_return': round(expected_return * 100, 2),  # As percentage
                    'horizon': horizon,
                    'model_accuracy': round(model_info['accuracy'] * 100, 1)
                }
                
            except Exception as e:
                logger.error(f"Prediction failed for {asset}: {e}")
                continue
        
        self.predictions = predictions
        return predictions
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return self.training_status.copy()
    
    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        return {
            'is_trained': self.is_trained,
            'num_models': len(self.models),
            'assets': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'models': {
                asset: {
                    'accuracy': info['accuracy'],
                    'trained_at': info['trained_at']
                }
                for asset, info in self.models.items()
            }
        }


# Singleton instance
multi_asset_predictor = MultiAssetPredictor()
