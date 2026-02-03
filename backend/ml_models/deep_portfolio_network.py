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
        
        try:
            # Prepare latest data
            X, _ = self._prepare_training_data(price_data, returns_data)
            
            if len(X) == 0:
                # Return equal weights if no data
                result = {}
                for asset in self.asset_names:
                    result[asset] = 1.0 / len(self.asset_names)
                return result
            
            # Check if input shape matches model expectation
            expected_shape = self.model.input_shape
            if X.shape[1] != expected_shape[1] or X.shape[2] != expected_shape[2]:
                logger.warning(f"Input shape mismatch: got {X.shape}, expected {expected_shape}")
                # Return equal weights on shape mismatch
                result = {}
                for asset in self.asset_names:
                    result[asset] = 1.0 / len(self.asset_names)
                return result
            
            # Get prediction for latest timestep
            latest_X = X[-1:].reshape(1, X.shape[1], X.shape[2])
            weights = self.model.predict(latest_X, verbose=0)[0]
            
            # Map weights to asset names
            result = {}
            for i, asset in enumerate(self.asset_names):
                if i < len(weights):
                    result[asset] = float(weights[i])
            
            return result
            
        except Exception as e:
            logger.error(f"Deep Learning prediction error: {e}")
            # Return equal weights on error
            result = {}
            for asset in self.asset_names:
                result[asset] = 1.0 / len(self.asset_names)
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
    
    def save(self, save_dir: str = "/app/backend/saved_models/portfolio") -> Dict:
        """
        Save the trained model to disk
        
        Args:
            save_dir: Directory to save the model
        
        Returns:
            Dictionary with save status and path
        """
        import os
        import json
        import joblib
        
        if not self.is_trained or self.model is None:
            return {'status': 'error', 'message': 'No trained model to save'}
        
        try:
            # Create directory if not exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            model_name = f"deep_portfolio_{timestamp}"
            model_path = os.path.join(save_dir, model_name)
            
            # Save Keras model
            self.model.save(f"{model_path}.keras")
            
            # Save metadata and scaler
            metadata = {
                'model_type': 'deep_portfolio_network',
                'n_assets': self.n_assets,
                'lookback': self.lookback,
                'feature_dim': self.feature_dim,
                'asset_names': self.asset_names,
                'created_at': timestamp,
                'model_params': self.model.count_params()
            }
            
            with open(f"{model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if self.scaler is not None:
                joblib.dump(self.scaler, f"{model_path}_scaler.joblib")
            
            logger.info(f"Deep Portfolio Network saved to {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'model_name': model_name,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving Deep Portfolio Network: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def load(self, model_path: str) -> Dict:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model (without extension)
        
        Returns:
            Dictionary with load status
        """
        import os
        import json
        import joblib
        import tensorflow as tf
        
        try:
            # Load Keras model
            keras_path = f"{model_path}.keras"
            if not os.path.exists(keras_path):
                return {'status': 'error', 'message': f'Model file not found: {keras_path}'}
            
            self.model = tf.keras.models.load_model(keras_path, compile=False)
            
            # Recompile with custom loss
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=self._sharpe_loss,
                metrics=['mae']
            )
            
            # Load metadata
            metadata_path = f"{model_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.n_assets = metadata.get('n_assets', 20)
                self.lookback = metadata.get('lookback', 30)
                self.feature_dim = metadata.get('feature_dim', 0)
                self.asset_names = metadata.get('asset_names', [])
            
            # Load scaler
            scaler_path = f"{model_path}_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            
            logger.info(f"Deep Portfolio Network loaded from {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'n_assets': self.n_assets,
                'asset_names': self.asset_names
            }
            
        except Exception as e:
            logger.error(f"Error loading Deep Portfolio Network: {e}")
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def list_saved_models(save_dir: str = "/app/backend/saved_models/portfolio") -> List[Dict]:
        """
        List all saved Deep Portfolio models
        
        Returns:
            List of model metadata dictionaries
        """
        import os
        import json
        
        models = []
        
        if not os.path.exists(save_dir):
            return models
        
        for filename in os.listdir(save_dir):
            if filename.startswith("deep_portfolio_") and filename.endswith("_metadata.json"):
                try:
                    with open(os.path.join(save_dir, filename), 'r') as f:
                        metadata = json.load(f)
                    
                    model_name = filename.replace("_metadata.json", "")
                    metadata['model_name'] = model_name
                    metadata['model_path'] = os.path.join(save_dir, model_name)
                    models.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {filename}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    @staticmethod
    def delete_model(model_path: str) -> Dict:
        """
        Delete a saved model
        
        Args:
            model_path: Path to the model (without extension)
        
        Returns:
            Dictionary with delete status
        """
        import os
        
        try:
            files_deleted = []
            
            # Delete all related files
            extensions = ['.keras', '_metadata.json', '_scaler.joblib']
            for ext in extensions:
                filepath = f"{model_path}{ext}"
                if os.path.exists(filepath):
                    os.remove(filepath)
                    files_deleted.append(filepath)
            
            if files_deleted:
                logger.info(f"Deleted Deep Portfolio model: {model_path}")
                return {
                    'status': 'success',
                    'files_deleted': files_deleted
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No files found to delete'
                }
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return {'status': 'error', 'message': str(e)}


# Singleton instance
deep_portfolio_network = DeepPortfolioNetwork()
