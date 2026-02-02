"""
Deep Learning Portfolio Network
A neural network that directly outputs portfolio allocation weights
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DeepPortfolioNetwork:
    """
    Deep Learning model that outputs optimal portfolio weights directly.
    Uses a neural network trained to maximize Sharpe ratio.
    """
    
    def __init__(self, n_assets: int = 20, lookback: int = 30):
        self.n_assets = n_assets
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.asset_names: List[str] = []
        self.is_trained = False
        self.training_history = None
        self.feature_dim = 0
        
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the portfolio allocation network"""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, Dropout, BatchNormalization,
            Concatenate, Attention, GlobalAveragePooling1D, Reshape
        )
        
        # Input: (timesteps, features_per_asset * n_assets)
        inputs = Input(shape=input_shape, name='market_data')
        
        # Shared LSTM encoder for temporal patterns
        x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(64, return_sequences=True, name='lstm_2')(x)
        x = Dropout(0.2)(x)
        
        # Attention mechanism to focus on important time steps
        attention = Attention(name='attention')([x, x])
        x = GlobalAveragePooling1D()(attention)
        
        # Dense layers for portfolio decision
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu', name='dense_2')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu', name='dense_3')(x)
        
        # Output: Portfolio weights (softmax ensures they sum to 1)
        outputs = Dense(self.n_assets, activation='softmax', name='portfolio_weights')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='DeepPortfolioNetwork')
        
        return model
    
    def _sharpe_loss(self, y_true, y_pred):
        """
        Custom loss function: Negative Sharpe Ratio
        y_true: Future returns for each asset (batch, n_assets)
        y_pred: Predicted weights (batch, n_assets)
        """
        import tensorflow as tf
        
        # Portfolio return = sum(weights * asset_returns)
        portfolio_returns = tf.reduce_sum(y_pred * y_true, axis=1)
        
        # Calculate mean and std of portfolio returns
        mean_return = tf.reduce_mean(portfolio_returns)
        std_return = tf.math.reduce_std(portfolio_returns) + 1e-6  # Avoid division by zero
        
        # Negative Sharpe ratio (we minimize this)
        risk_free_rate = 0.0  # Simplified
        sharpe = (mean_return - risk_free_rate) / std_return
        
        # Add diversification penalty (encourage spreading weights)
        concentration = tf.reduce_sum(tf.square(y_pred), axis=1)
        concentration_penalty = tf.reduce_mean(concentration) * 0.1
        
        return -sharpe + concentration_penalty
    
    def _prepare_training_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from price and returns data
        
        Returns:
            X: (samples, lookback, features)
            y: (samples, n_assets) - future returns
        """
        from sklearn.preprocessing import RobustScaler
        
        self.asset_names = list(price_data.keys())
        self.n_assets = len(self.asset_names)
        
        # Align all data to common timestamps
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < self.lookback + 10:
            logger.warning(f"Insufficient data: {len(returns_df)} rows")
            return np.array([]), np.array([])
        
        # Build feature matrix for each asset
        feature_matrices = []
        
        for asset in self.asset_names:
            if asset not in price_data:
                continue
                
            df = price_data[asset].copy()
            
            # Basic features
            features = pd.DataFrame(index=df.index)
            features['return'] = df['close'].pct_change()
            features['log_return'] = np.log(df['close'] / df['close'].shift(1))
            features['volatility'] = features['return'].rolling(10).std()
            features['momentum'] = df['close'] / df['close'].shift(5) - 1
            features['volume_change'] = df['volume'].pct_change()
            
            # Normalize
            features = features.dropna()
            feature_matrices.append(features)
        
        if not feature_matrices:
            return np.array([]), np.array([])
        
        # Align all features to common index
        common_index = feature_matrices[0].index
        for fm in feature_matrices[1:]:
            common_index = common_index.intersection(fm.index)
        
        # Stack features for all assets
        aligned_features = []
        for fm in feature_matrices:
            aligned_features.append(fm.loc[common_index].values)
        
        # Shape: (timesteps, n_assets * n_features)
        all_features = np.hstack(aligned_features)
        self.feature_dim = all_features.shape[1]
        
        # Scale features
        self.scaler = RobustScaler()
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        # Create sequences
        X, y = [], []
        
        # Align returns to same index
        returns_aligned = returns_df.loc[common_index]
        
        for i in range(self.lookback, len(all_features_scaled) - 1):
            # Input: lookback window of features
            X.append(all_features_scaled[i-self.lookback:i])
            
            # Target: next period returns for all assets
            future_returns = []
            for asset in self.asset_names:
                if asset in returns_aligned.columns:
                    ret = returns_aligned[asset].iloc[i] if i < len(returns_aligned) else 0
                    future_returns.append(ret if not np.isnan(ret) else 0)
                else:
                    future_returns.append(0)
            y.append(future_returns)
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def train(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the deep portfolio network
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Prepare data
        X, y = self._prepare_training_data(price_data, returns_data)
        
        if len(X) < 50:
            return {
                'status': 'failed',
                'reason': 'insufficient_data',
                'samples': len(X)
            }
        
        # Build model
        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Compile with custom loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._sharpe_loss,
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        # Calculate final metrics
        val_loss = min(history.history.get('val_loss', [0]))
        
        logger.info(f"Deep Portfolio Network trained: val_loss={val_loss:.4f}")
        
        return {
            'status': 'success',
            'epochs_run': len(history.history['loss']),
            'final_loss': float(val_loss),
            'n_assets': self.n_assets,
            'samples': len(X)
        }
    
    def predict_weights(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Predict optimal portfolio weights
        
        Returns:
            Dictionary mapping asset names to weights
        """
        if not self.is_trained or self.model is None:
            return {}
        
        # Prepare latest data
        X, _ = self._prepare_training_data(price_data, returns_data)
        
        if len(X) == 0:
            return {}
        
        # Get prediction for latest timestep
        latest_X = X[-1:].reshape(1, X.shape[1], X.shape[2])
        weights = self.model.predict(latest_X, verbose=0)[0]
        
        # Map weights to asset names
        result = {}
        for i, asset in enumerate(self.asset_names):
            if i < len(weights):
                result[asset] = float(weights[i])
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'n_assets': self.n_assets,
            'lookback': self.lookback,
            'feature_dim': self.feature_dim,
            'asset_names': self.asset_names,
            'model_params': self.model.count_params() if self.model else 0
        }


# Singleton instance
deep_portfolio_network = DeepPortfolioNetwork()
