"""
Advanced Model Architectures with Multiple Network Types
- LSTM (Bi-directional)
- GRU
- Transformer
- CNN + LSTM
- Ensemble (LSTM + XGBoost + RandomForest)
- Late Fusion Multi-Timeframe
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Model save directory
MODEL_DIR = Path("/app/backend/saved_models")
MODEL_DIR.mkdir(exist_ok=True)


class ProgressCallback(Callback):
    """Callback to report training progress"""
    
    def __init__(self, status_dict: Dict):
        super().__init__()
        self.status = status_dict
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.status['current_epoch'] = epoch + 1
        self.status['current_loss'] = float(logs.get('loss', 0))
        self.status['current_accuracy'] = float(logs.get('accuracy', 0))
        
        history_entry = {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0)),
            'auc': float(logs.get('auc', 0))
        }
        
        if 'history' not in self.status:
            self.status['history'] = []
        self.status['history'].append(history_entry)


def build_lstm_model(input_shape: Tuple[int, int], 
                    num_lstm_layers: int = 2,
                    lstm_units: List[int] = [128, 64],
                    num_dense_layers: int = 2,
                    dense_units: List[int] = [64, 32],
                    dropout_rate: float = 0.3,
                    use_attention: bool = True,
                    use_batch_norm: bool = True,
                    learning_rate: float = 0.001) -> Model:
    """Build Bi-directional LSTM model"""
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # LSTM layers
    for i in range(num_lstm_layers):
        units = lstm_units[i] if i < len(lstm_units) else 64
        return_sequences = (i < num_lstm_layers - 1) or use_attention
        
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_sequences, 
                       kernel_regularizer=keras.regularizers.l2(0.01)),
            name=f'bilstm_{i+1}'
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(x)
    
    # Attention layer
    if use_attention:
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, name='attention'
        )(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization(name='ln_attention')(x)
        x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        units = dense_units[i] if i < len(dense_units) else 32
        x = layers.Dense(units, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        name=f'dense_{i+1}')(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='LSTM_Model')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def build_gru_model(input_shape: Tuple[int, int],
                   num_gru_layers: int = 2,
                   gru_units: List[int] = [128, 64],
                   num_dense_layers: int = 2,
                   dense_units: List[int] = [64, 32],
                   dropout_rate: float = 0.3,
                   use_attention: bool = True,
                   use_batch_norm: bool = True,
                   learning_rate: float = 0.001) -> Model:
    """Build GRU model"""
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # GRU layers
    for i in range(num_gru_layers):
        units = gru_units[i] if i < len(gru_units) else 64
        return_sequences = (i < num_gru_layers - 1) or use_attention
        
        x = layers.Bidirectional(
            layers.GRU(units, return_sequences=return_sequences,
                      kernel_regularizer=keras.regularizers.l2(0.01)),
            name=f'bigru_{i+1}'
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_gru_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_gru_{i+1}')(x)
    
    # Attention
    if use_attention:
        attention = layers.MultiHeadAttention(
            num_heads=4, key_dim=32, name='attention'
        )(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization(name='ln_attention')(x)
        x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        units = dense_units[i] if i < len(dense_units) else 32
        x = layers.Dense(units, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        name=f'dense_{i+1}')(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='GRU_Model')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def build_transformer_model(input_shape: Tuple[int, int],
                           num_transformer_blocks: int = 2,
                           num_heads: int = 4,
                           ff_dim: int = 128,
                           num_dense_layers: int = 2,
                           dense_units: List[int] = [64, 32],
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.001) -> Model:
    """Build Transformer model"""
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # Positional encoding (learnable)
    positions = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(
        tf.range(start=0, limit=input_shape[0], delta=1)
    )
    x = x + positions
    
    # Transformer blocks
    for i in range(num_transformer_blocks):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1] // num_heads,
            name=f'mha_{i+1}'
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(name=f'ln1_{i+1}')(x + attn_output)
        
        # Feed-forward
        ff_output = layers.Dense(ff_dim, activation='relu', name=f'ff1_{i+1}')(x)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        ff_output = layers.Dense(input_shape[1], name=f'ff2_{i+1}')(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(name=f'ln2_{i+1}')(x + ff_output)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        units = dense_units[i] if i < len(dense_units) else 32
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='Transformer_Model')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def build_cnn_lstm_model(input_shape: Tuple[int, int],
                        num_conv_layers: int = 2,
                        conv_filters: List[int] = [64, 128],
                        kernel_size: int = 3,
                        num_lstm_layers: int = 1,
                        lstm_units: List[int] = [64],
                        num_dense_layers: int = 2,
                        dense_units: List[int] = [64, 32],
                        dropout_rate: float = 0.3,
                        use_batch_norm: bool = True,
                        learning_rate: float = 0.001) -> Model:
    """Build CNN + LSTM hybrid model"""
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs
    
    # CNN layers
    for i in range(num_conv_layers):
        filters = conv_filters[i] if i < len(conv_filters) else 64
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                         name=f'conv_{i+1}')(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
    
    # LSTM layers
    for i in range(num_lstm_layers):
        units = lstm_units[i] if i < len(lstm_units) else 64
        return_sequences = i < num_lstm_layers - 1
        x = layers.LSTM(units, return_sequences=return_sequences,
                       name=f'lstm_{i+1}')(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_lstm_{i+1}')(x)
    
    # Dense layers
    for i in range(num_dense_layers):
        units = dense_units[i] if i < len(dense_units) else 32
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='CNN_LSTM_Model')
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


class EnsembleModel:
    """
    Ensemble model combining:
    - LSTM (deep learning)
    - XGBoost (gradient boosting)
    - Random Forest (bagging)
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict = None):
        self.config = config or {}
        self.input_shape = input_shape
        
        # Neural network component
        self.nn_model = build_lstm_model(
            input_shape,
            num_lstm_layers=self.config.get('num_lstm_layers', 2),
            lstm_units=self.config.get('lstm_units', [64, 32]),
            num_dense_layers=self.config.get('num_dense_layers', 1),
            dense_units=self.config.get('dense_units', [32]),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            use_attention=self.config.get('use_attention', True),
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        
        # XGBoost component
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        
        # Random Forest component
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        # Ensemble weights (can be tuned)
        self.weights = self.config.get('ensemble_weights', [0.5, 0.3, 0.2])
        
        self.is_trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           validation_data: Tuple = None,
           epochs: int = 50,
           batch_size: int = 32,
           callbacks: List = None) -> Dict:
        """Train all ensemble components"""
        
        history = {'nn': None, 'xgb': None, 'rf': None}
        
        # Train neural network
        logger.info("Training LSTM component...")
        nn_history = self.nn_model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        history['nn'] = nn_history.history
        
        # Flatten features for tree-based models
        X_flat = X.reshape(X.shape[0], -1)
        
        # Train XGBoost
        logger.info("Training XGBoost component...")
        if validation_data:
            X_val_flat = validation_data[0].reshape(validation_data[0].shape[0], -1)
            self.xgb_model.fit(
                X_flat, y,
                eval_set=[(X_val_flat, validation_data[1])],
                verbose=False
            )
        else:
            self.xgb_model.fit(X_flat, y)
        
        # Train Random Forest
        logger.info("Training Random Forest component...")
        self.rf_model.fit(X_flat, y)
        
        self.is_trained = True
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction with weighted voting"""
        if not self.is_trained:
            return np.array([0.5] * len(X))
        
        # Neural network prediction
        nn_pred = self.nn_model.predict(X, verbose=0).flatten()
        
        # Tree-based predictions
        X_flat = X.reshape(X.shape[0], -1)
        xgb_pred = self.xgb_model.predict_proba(X_flat)[:, 1]
        rf_pred = self.rf_model.predict_proba(X_flat)[:, 1]
        
        # Weighted ensemble
        ensemble_pred = (
            self.weights[0] * nn_pred +
            self.weights[1] * xgb_pred +
            self.weights[2] * rf_pred
        )
        
        return ensemble_pred
    
    def save(self, path: str):
        """Save all model components"""
        os.makedirs(path, exist_ok=True)
        
        # Save neural network
        self.nn_model.save(os.path.join(path, 'nn_model.keras'))
        
        # Save XGBoost
        self.xgb_model.save_model(os.path.join(path, 'xgb_model.json'))
        
        # Save Random Forest
        joblib.dump(self.rf_model, os.path.join(path, 'rf_model.joblib'))
        
        # Save config
        joblib.dump({
            'config': self.config,
            'weights': self.weights,
            'input_shape': self.input_shape
        }, os.path.join(path, 'ensemble_config.joblib'))
        
        logger.info(f"Ensemble model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load ensemble model from disk"""
        config_data = joblib.load(os.path.join(path, 'ensemble_config.joblib'))
        
        model = cls(config_data['input_shape'], config_data['config'])
        model.weights = config_data['weights']
        
        # Load components
        model.nn_model = keras.models.load_model(os.path.join(path, 'nn_model.keras'))
        model.xgb_model.load_model(os.path.join(path, 'xgb_model.json'))
        model.rf_model = joblib.load(os.path.join(path, 'rf_model.joblib'))
        
        model.is_trained = True
        
        logger.info(f"Ensemble model loaded from {path}")
        return model


def build_model(network_type: str, input_shape: Tuple[int, int], config: Dict) -> Model:
    """Factory function to build model based on network type"""
    
    # Import advanced architectures
    from ml_models.advanced_architectures import build_advanced_model
    
    builders = {
        'lstm': lambda: build_lstm_model(
            input_shape,
            num_lstm_layers=config.get('num_lstm_layers', 2),
            lstm_units=config.get('lstm_units', [128, 64]),
            num_dense_layers=config.get('num_dense_layers', 2),
            dense_units=config.get('dense_units', [64, 32]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_attention=config.get('use_attention', True),
            use_batch_norm=config.get('use_batch_norm', True),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'gru': lambda: build_gru_model(
            input_shape,
            num_gru_layers=config.get('num_lstm_layers', 2),
            gru_units=config.get('lstm_units', [128, 64]),
            num_dense_layers=config.get('num_dense_layers', 2),
            dense_units=config.get('dense_units', [64, 32]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_attention=config.get('use_attention', True),
            use_batch_norm=config.get('use_batch_norm', True),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'transformer': lambda: build_transformer_model(
            input_shape,
            num_transformer_blocks=config.get('num_lstm_layers', 2),
            num_heads=config.get('num_heads', 4),
            ff_dim=config.get('ff_dim', 128),
            num_dense_layers=config.get('num_dense_layers', 2),
            dense_units=config.get('dense_units', [64, 32]),
            dropout_rate=config.get('dropout_rate', 0.3),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'cnn_lstm': lambda: build_cnn_lstm_model(
            input_shape,
            num_conv_layers=config.get('num_conv_layers', 2),
            conv_filters=config.get('conv_filters', [64, 128]),
            num_lstm_layers=config.get('num_lstm_layers', 1),
            lstm_units=config.get('lstm_units', [64]),
            num_dense_layers=config.get('num_dense_layers', 2),
            dense_units=config.get('dense_units', [64, 32]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        # Advanced Models
        'tft': lambda: build_advanced_model('tft', input_shape, config),
        'multi_task': lambda: build_advanced_model('multi_task', input_shape, config),
        'gnn': lambda: build_advanced_model('gnn', input_shape, config),
        'multi_tf_attention': lambda: build_advanced_model('multi_tf_attention', input_shape, config),
    }
    
    if network_type not in builders:
        logger.warning(f"Unknown network type {network_type}, defaulting to LSTM")
        network_type = 'lstm'
    
    return builders[network_type]()


def save_model(model: Model, symbol: str, network_type: str, 
               metrics: Dict = None, config: Dict = None):
    """Save trained model to disk"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol.replace('/', '_')}_{network_type}_{timestamp}"
    model_path = MODEL_DIR / model_name
    
    os.makedirs(model_path, exist_ok=True)
    
    # Save Keras model
    model.save(model_path / 'model.keras')
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'network_type': network_type,
        'timestamp': timestamp,
        'metrics': metrics or {},
        'config': config or {}
    }
    joblib.dump(metadata, model_path / 'metadata.joblib')
    
    logger.info(f"Model saved to {model_path}")
    
    return str(model_path)


def load_model(model_path: str) -> Tuple[Model, Dict]:
    """Load model from disk"""
    model_path = Path(model_path)
    
    model = keras.models.load_model(model_path / 'model.keras')
    metadata = joblib.load(model_path / 'metadata.joblib')
    
    logger.info(f"Model loaded from {model_path}")
    
    return model, metadata


def list_saved_models() -> List[Dict]:
    """List all saved models"""
    models = []
    
    for model_dir in MODEL_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / 'metadata.joblib'
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                metadata['path'] = str(model_dir)
                models.append(metadata)
    
    # Sort by timestamp descending
    models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return models
