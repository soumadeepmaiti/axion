"""
Hybrid Deep Learning Model for Crypto Trading
Multi-Input Late Fusion Architecture
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json
import os

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using mock model")


class HybridCryptoModel:
    """
    Multi-Input Hybrid Deep Learning Model
    
    Branch A: Bi-LSTM for micro timeframes (1m, 5m, 10m)
    Branch B: Attention-based GRU for macro trends (1h, 1d)
    Branch C: Sentiment embedding from FinBERT
    
    Late Fusion: Concatenate all branches -> Dense -> Output
    """
    
    def __init__(
        self,
        micro_input_shape: Tuple[int, int] = (50, 15),  # (sequence_length, features)
        macro_input_shape: Tuple[int, int] = (24, 10),  # (sequence_length, features)
        sentiment_input_shape: Tuple[int,] = (4,),  # sentiment features
        lstm_units: int = 64,
        gru_units: int = 32,
        dense_units: int = 64,
        dropout_rate: float = 0.3
    ):
        self.micro_input_shape = micro_input_shape
        self.macro_input_shape = macro_input_shape
        self.sentiment_input_shape = sentiment_input_shape
        self.lstm_units = lstm_units
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_micro_branch(self, input_layer):
        """Branch A: Bi-LSTM for short-term price action"""
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(input_layer)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=False)
        )(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def build_macro_branch(self, input_layer):
        """Branch B: Attention-based GRU for macro trends"""
        # GRU layer
        x = layers.GRU(self.gru_units, return_sequences=True)(input_layer)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Simple attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(self.gru_units)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def build_sentiment_branch(self, input_layer):
        """Branch C: Sentiment embedding processor"""
        x = layers.Dense(16, activation='relu')(input_layer)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(8, activation='relu')(x)
        return x
    
    def build_model(self):
        """Build the complete multi-input hybrid model"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, model will use mock predictions")
            return None
            
        # Input layers
        micro_input = layers.Input(shape=self.micro_input_shape, name='micro_input')
        macro_input = layers.Input(shape=self.macro_input_shape, name='macro_input')
        sentiment_input = layers.Input(shape=self.sentiment_input_shape, name='sentiment_input')
        
        # Build branches
        micro_features = self.build_micro_branch(micro_input)
        macro_features = self.build_macro_branch(macro_input)
        sentiment_features = self.build_sentiment_branch(sentiment_input)
        
        # Late Fusion: Concatenate all branches
        fusion = layers.Concatenate()([micro_features, macro_features, sentiment_features])
        
        # Dense hidden layers
        x = layers.Dense(self.dense_units, activation='relu')(fusion)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.dense_units // 2, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        # Output layers
        # Price direction (0 = down, 1 = up)
        direction_output = layers.Dense(1, activation='sigmoid', name='direction')(x)
        
        # Confidence score
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        # Build model
        self.model = Model(
            inputs=[micro_input, macro_input, sentiment_input],
            outputs=[direction_output, confidence_output]
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'direction': 'binary_crossentropy',
                'confidence': 'mse'
            },
            loss_weights={'direction': 1.0, 'confidence': 0.3},
            metrics={
                'direction': ['accuracy', keras.metrics.AUC(name='auc')],
                'confidence': ['mae']
            }
        )
        
        logger.info("Model built successfully")
        logger.info(f"Model summary: {self.model.summary()}")
        
        return self.model
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary"""
        if self.model is None:
            return {
                "status": "not_built",
                "micro_input_shape": self.micro_input_shape,
                "macro_input_shape": self.macro_input_shape,
                "sentiment_input_shape": self.sentiment_input_shape,
                "total_params": 0
            }
            
        return {
            "status": "built",
            "micro_input_shape": self.micro_input_shape,
            "macro_input_shape": self.macro_input_shape,
            "sentiment_input_shape": self.sentiment_input_shape,
            "lstm_units": self.lstm_units,
            "gru_units": self.gru_units,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "total_params": self.model.count_params() if TF_AVAILABLE else 0,
            "is_trained": self.is_trained
        }
    
    def prepare_inputs(
        self,
        micro_features: np.ndarray,
        macro_features: np.ndarray,
        sentiment_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and pad inputs to expected shapes"""
        
        # Pad micro features
        if len(micro_features) < self.micro_input_shape[0]:
            pad_size = self.micro_input_shape[0] - len(micro_features)
            micro_features = np.pad(
                micro_features,
                ((pad_size, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            micro_features = micro_features[-self.micro_input_shape[0]:]
            
        # Ensure correct feature dimension
        if micro_features.shape[1] < self.micro_input_shape[1]:
            pad_cols = self.micro_input_shape[1] - micro_features.shape[1]
            micro_features = np.pad(
                micro_features,
                ((0, 0), (0, pad_cols)),
                mode='constant'
            )
        elif micro_features.shape[1] > self.micro_input_shape[1]:
            micro_features = micro_features[:, :self.micro_input_shape[1]]
            
        # Pad macro features
        if len(macro_features) < self.macro_input_shape[0]:
            pad_size = self.macro_input_shape[0] - len(macro_features)
            macro_features = np.pad(
                macro_features,
                ((pad_size, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            macro_features = macro_features[-self.macro_input_shape[0]:]
            
        if macro_features.shape[1] < self.macro_input_shape[1]:
            pad_cols = self.macro_input_shape[1] - macro_features.shape[1]
            macro_features = np.pad(
                macro_features,
                ((0, 0), (0, pad_cols)),
                mode='constant'
            )
        elif macro_features.shape[1] > self.macro_input_shape[1]:
            macro_features = macro_features[:, :self.macro_input_shape[1]]
            
        # Ensure sentiment features
        if len(sentiment_features) < self.sentiment_input_shape[0]:
            pad_size = self.sentiment_input_shape[0] - len(sentiment_features)
            sentiment_features = np.pad(
                sentiment_features,
                (0, pad_size),
                mode='constant'
            )
        else:
            sentiment_features = sentiment_features[:self.sentiment_input_shape[0]]
            
        return (
            micro_features.reshape(1, *self.micro_input_shape),
            macro_features.reshape(1, *self.macro_input_shape),
            sentiment_features.reshape(1, *self.sentiment_input_shape)
        )
    
    def predict(
        self,
        micro_features: np.ndarray,
        macro_features: np.ndarray,
        sentiment_features: np.ndarray
    ) -> Dict:
        """Make prediction using the hybrid model"""
        
        micro_input, macro_input, sentiment_input = self.prepare_inputs(
            micro_features, macro_features, sentiment_features
        )
        
        if self.model is None or not TF_AVAILABLE:
            # Mock prediction
            direction = np.random.random()
            confidence = np.random.random() * 0.3 + 0.5
            return {
                "direction": int(direction > 0.5),
                "direction_probability": round(float(direction), 4),
                "confidence": round(float(confidence), 4),
                "model_status": "mock"
            }
        
        direction_pred, confidence_pred = self.model.predict(
            [micro_input, macro_input, sentiment_input],
            verbose=0
        )
        
        return {
            "direction": int(direction_pred[0][0] > 0.5),
            "direction_probability": round(float(direction_pred[0][0]), 4),
            "confidence": round(float(confidence_pred[0][0]), 4),
            "model_status": "trained" if self.is_trained else "untrained"
        }
    
    def save(self, path: str):
        """Save model weights"""
        if self.model is not None and TF_AVAILABLE:
            self.model.save_weights(path)
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights"""
        if os.path.exists(path) and TF_AVAILABLE:
            if self.model is None:
                self.build_model()
            self.model.load_weights(path)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")


class MockHybridModel:
    """Mock model for when TensorFlow is not available"""
    
    def __init__(self):
        self.is_trained = False
        
    def predict(self, micro_features, macro_features, sentiment_features) -> Dict:
        """Generate mock predictions based on input features"""
        # Use feature statistics to generate semi-realistic predictions
        micro_mean = np.mean(micro_features) if len(micro_features) > 0 else 0
        sentiment_score = sentiment_features[0] if len(sentiment_features) > 0 else 0
        
        # Combine signals
        combined_signal = (micro_mean * 0.3 + sentiment_score * 0.7)
        direction_prob = 1 / (1 + np.exp(-combined_signal))  # Sigmoid
        
        return {
            "direction": int(direction_prob > 0.5),
            "direction_probability": round(float(direction_prob), 4),
            "confidence": round(float(abs(direction_prob - 0.5) * 2), 4),
            "model_status": "mock"
        }
    
    def get_model_summary(self) -> Dict:
        return {
            "status": "mock",
            "message": "TensorFlow not available, using mock predictions"
        }


# Create model instance
if TF_AVAILABLE:
    hybrid_model = HybridCryptoModel()
else:
    hybrid_model = MockHybridModel()
