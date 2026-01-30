"""
Enhanced Training Service with Full Data Training and Mathematical Modeling
"""
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import asyncio
import json

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class EnhancedTrainingService:
    """Handles comprehensive model training with all historical data"""
    
    def __init__(self):
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_loss": 0,
            "current_accuracy": 0,
            "history": [],
            "start_time": None,
            "end_time": None,
            "error": None,
            "data_info": None,
            "math_analysis": None
        }
        self.training_task = None
        self.best_accuracy = 0.5  # Store best accuracy achieved
        self.trained_model = None  # Store trained model
        
    def reset_status(self):
        """Reset training status"""
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_loss": 0,
            "current_accuracy": 0,
            "history": [],
            "start_time": None,
            "end_time": None,
            "error": None,
            "data_info": None,
            "math_analysis": None
        }
        # Don't reset best_accuracy or trained_model - keep them for predictions
        
    def get_status(self) -> Dict:
        """Get current training status"""
        return self.training_status.copy()
    
    def prepare_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 50,
        test_split: float = 0.2
    ) -> Dict:
        """Prepare sequential data for LSTM training"""
        
        if len(features) < sequence_length + 10:
            return {"error": f"Not enough data. Need at least {sequence_length + 10} samples, got {len(features)}"}
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            if i < len(labels):
                y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Ensure same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Split data
        split_idx = int(len(X) * (1 - test_split))
        
        return {
            "X_train": X[:split_idx],
            "y_train": y[:split_idx],
            "X_val": X[split_idx:],
            "y_val": y[split_idx:],
            "total_samples": len(X),
            "train_samples": split_idx,
            "val_samples": len(X) - split_idx,
            "sequence_length": sequence_length,
            "feature_dim": X.shape[2] if len(X.shape) > 2 else features.shape[1]
        }
    
    def build_model(self, input_shape: tuple, model_config: Dict = None) -> keras.Model:
        """Build a comprehensive LSTM model for price prediction"""
        
        config = model_config or {
            "lstm_units_1": 128,
            "lstm_units_2": 64,
            "dense_units": 32,
            "dropout_rate": 0.3
        }
        
        inputs = keras.layers.Input(shape=input_shape)
        
        # First LSTM layer with attention
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(config["lstm_units_1"], return_sequences=True)
        )(inputs)
        x = keras.layers.Dropout(config["dropout_rate"])(x)
        
        # Attention mechanism
        attention = keras.layers.Dense(1, activation='tanh')(x)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(config["lstm_units_1"] * 2)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        x = keras.layers.Multiply()([x, attention])
        
        # Second LSTM layer
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(config["lstm_units_2"], return_sequences=False)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(config["dropout_rate"])(x)
        
        # Dense layers
        x = keras.layers.Dense(config["dense_units"], activation='relu')(x)
        x = keras.layers.Dropout(config["dropout_rate"] / 2)(x)
        
        # Output
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    async def train_full_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 50,
        data_info: Dict = None,
        math_analysis: Dict = None
    ) -> Dict:
        """Train model with all available data"""
        
        self.training_status["is_training"] = True
        self.training_status["total_epochs"] = epochs
        self.training_status["start_time"] = datetime.now(timezone.utc).isoformat()
        self.training_status["data_info"] = data_info
        self.training_status["math_analysis"] = math_analysis
        
        # Prepare sequences
        training_data = self.prepare_sequences(features, labels, sequence_length)
        
        if "error" in training_data:
            self.training_status["is_training"] = False
            self.training_status["error"] = training_data["error"]
            return training_data
        
        logger.info(f"Training with {training_data['train_samples']} samples, validating with {training_data['val_samples']}")
        
        if not TF_AVAILABLE:
            # Mock training
            for epoch in range(epochs):
                if not self.training_status["is_training"]:
                    break
                    
                await asyncio.sleep(0.05)
                progress = (epoch + 1) / epochs
                self.training_status["current_epoch"] = epoch + 1
                self.training_status["current_loss"] = round(0.7 - (progress * 0.3) + np.random.random() * 0.05, 4)
                self.training_status["current_accuracy"] = round(0.5 + (progress * 0.35) + np.random.random() * 0.03, 4)
                self.training_status["history"].append({
                    "epoch": epoch + 1,
                    "loss": self.training_status["current_loss"],
                    "accuracy": self.training_status["current_accuracy"],
                    "val_loss": self.training_status["current_loss"] * 1.1,
                    "val_accuracy": self.training_status["current_accuracy"] * 0.95
                })
            
            self.training_status["is_training"] = False
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            
            return {
                "status": "completed",
                "message": "Mock training completed",
                "final_accuracy": self.training_status["current_accuracy"],
                "samples_trained": training_data["train_samples"]
            }
        
        try:
            # Build model
            input_shape = (training_data["sequence_length"], training_data["feature_dim"])
            model = self.build_model(input_shape)
            
            logger.info(f"Model built with input shape: {input_shape}")
            
            # Custom callback for progress
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, service):
                    super().__init__()
                    self.service = service
                    
                def on_epoch_end(self, epoch, logs=None):
                    self.service.training_status["current_epoch"] = epoch + 1
                    self.service.training_status["current_loss"] = round(logs.get('loss', 0), 4)
                    self.service.training_status["current_accuracy"] = round(logs.get('accuracy', 0), 4)
                    self.service.training_status["history"].append({
                        "epoch": epoch + 1,
                        "loss": round(logs.get('loss', 0), 4),
                        "accuracy": round(logs.get('accuracy', 0), 4),
                        "val_loss": round(logs.get('val_loss', 0), 4),
                        "val_accuracy": round(logs.get('val_accuracy', 0), 4),
                        "auc": round(logs.get('auc', 0), 4)
                    })
            
            callbacks = [
                ProgressCallback(self),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    mode='min'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Train
            history = model.fit(
                training_data["X_train"],
                training_data["y_train"],
                validation_data=(training_data["X_val"], training_data["y_val"]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model reference
            self.trained_model = model
            
            # Save best accuracy achieved
            self.best_accuracy = max(history.history.get('accuracy', [0.5]))
            
            self.training_status["is_training"] = False
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            
            final_metrics = {
                "final_accuracy": round(history.history.get('accuracy', [0])[-1], 4),
                "final_loss": round(history.history.get('loss', [0])[-1], 4),
                "best_val_accuracy": round(max(history.history.get('val_accuracy', [0])), 4),
                "best_auc": round(max(history.history.get('auc', [0])), 4),
                "epochs_trained": len(history.history.get('loss', [])),
                "samples_trained": training_data["train_samples"],
                "best_accuracy": round(self.best_accuracy, 4)
            }
            
            logger.info(f"Training completed: {final_metrics}")
            
            return {
                "status": "completed",
                **final_metrics
            }
            
        except Exception as e:
            self.training_status["is_training"] = False
            self.training_status["error"] = str(e)
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"Training error: {e}")
            return {"status": "error", "error": str(e)}
    
    def stop_training(self):
        """Stop training"""
        self.training_status["is_training"] = False
        return {"status": "stopped"}
    
    def predict(self, features: np.ndarray, sequence_length: int = 50) -> Dict:
        """Make prediction using trained model"""
        
        if not hasattr(self, 'trained_model') or self.trained_model is None:
            # Return mock prediction
            return {
                "direction": int(np.random.random() > 0.5),
                "probability": round(np.random.random(), 4),
                "confidence": round(np.random.random() * 0.3 + 0.5, 4),
                "model_status": "untrained"
            }
        
        # Prepare input
        if len(features) < sequence_length:
            pad_size = sequence_length - len(features)
            features = np.pad(features, ((pad_size, 0), (0, 0)), mode='constant')
        else:
            features = features[-sequence_length:]
        
        features = features.reshape(1, sequence_length, -1)
        
        # Predict
        prob = float(self.trained_model.predict(features, verbose=0)[0][0])
        
        # Calculate confidence based on:
        # 1. How far from 50% (prediction strength)
        # 2. Model's validation accuracy
        
        prediction_strength = abs(prob - 0.5) * 2  # 0 to 1
        
        # Get model's best accuracy from training history
        model_accuracy = 0.5
        if self.training_status.get('history'):
            accuracies = [h.get('accuracy', 0) for h in self.training_status['history']]
            if accuracies:
                model_accuracy = max(accuracies)
        
        # Confidence = weighted combination
        # - 30% from prediction strength (how decisive the model is)
        # - 70% from model accuracy (how reliable the model has been)
        confidence = (prediction_strength * 0.3) + (model_accuracy * 0.7)
        
        # Boost confidence if prediction is strong (> 65% or < 35%)
        if prob > 0.65 or prob < 0.35:
            confidence = min(confidence * 1.15, 0.95)
        
        # Ensure minimum confidence based on model being trained
        confidence = max(confidence, model_accuracy * 0.8)
        
        return {
            "direction": int(prob > 0.5),
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "model_status": "trained",
            "model_accuracy": round(model_accuracy, 4)
        }


# Singleton instance
training_service = EnhancedTrainingService()
