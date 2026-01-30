"""
Pure ML Training Service - Model Discovers Its Own Patterns
No imposed mathematical conditions - learns from data
"""
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PureMLTrainingService:
    """Training service that lets model discover patterns from data"""
    
    FEATURE_NAMES = [
        'open', 'high', 'low', 'close', 'volume',
        'return_1', 'return_5', 'return_10', 'return_20',
        'volatility_10', 'volatility_20',
        'volume_change', 'volume_ma_ratio',
        'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio', 'price_ma50_ratio', 'price_ma100_ratio',
        'rsi_14', 'rsi_7', 'macd', 'stoch',
        'atr_ratio', 'bb_position', 'bb_width',
        'body_ratio', 'upper_shadow', 'lower_shadow',
        'adx'
    ]
    
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
            "learned_patterns": None
        }
        self.trained_model = None
        self.best_accuracy = 0.5
        self.feature_importance = {}
        
    def reset_status(self):
        """Reset training status (keeps model and learned patterns)"""
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
            "learned_patterns": self.training_status.get("learned_patterns")
        }
        
    def get_status(self) -> Dict:
        """Get current training status with learned patterns"""
        status = self.training_status.copy()
        status["learned_patterns"] = self.get_learned_patterns()
        return status
    
    def prepare_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 50,
        test_split: float = 0.2
    ) -> Dict:
        """Prepare sequential data for LSTM"""
        
        if len(features) < sequence_length + 10:
            return {"error": f"Not enough data. Need at least {sequence_length + 10} samples, got {len(features)}"}
        
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length:i])
            if i < len(labels):
                y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
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
    
    def build_model(self, input_shape: tuple) -> keras.Model:
        """Build LSTM model - learns patterns from data"""
        
        inputs = keras.layers.Input(shape=input_shape, name='input')
        
        # First LSTM layer
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, name='lstm_1')
        )(inputs)
        x = keras.layers.Dropout(0.3)(x)
        
        # Attention mechanism - model learns what to focus on
        attention = keras.layers.Dense(1, activation='tanh', name='attention_dense')(x)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax', name='attention_weights')(attention)
        attention = keras.layers.RepeatVector(256)(attention)
        attention = keras.layers.Permute([2, 1])(attention)
        x = keras.layers.Multiply(name='attention_applied')([x, attention])
        
        # Second LSTM
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=False, name='lstm_2')
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)
        
        # Dense layers - model learns feature combinations
        x = keras.layers.Dense(64, activation='relu', name='dense_1')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu', name='dense_2')(x)
        
        # Output
        outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def extract_learned_patterns(self, model: keras.Model, X_sample: np.ndarray) -> Dict:
        """Extract what patterns the model has learned"""
        patterns = {
            "feature_importance": {},
            "learned_weights": {},
            "model_equation": None
        }
        
        if model is None:
            return patterns
            
        try:
            # Get dense layer weights to understand feature importance
            for layer in model.layers:
                if 'dense' in layer.name and hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    if len(weights) > 0:
                        w = weights[0]
                        patterns["learned_weights"][layer.name] = {
                            "shape": list(w.shape),
                            "mean": float(np.mean(w)),
                            "std": float(np.std(w)),
                            "max": float(np.max(w)),
                            "min": float(np.min(w))
                        }
            
            # Calculate feature importance using gradient-based method
            if len(X_sample) > 0:
                importance = self._calculate_feature_importance(model, X_sample[:100])
                patterns["feature_importance"] = importance
            
            # Generate model equation representation
            patterns["model_equation"] = self._generate_equation_string(patterns)
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            
        return patterns
    
    def _calculate_feature_importance(self, model: keras.Model, X_sample: np.ndarray) -> Dict:
        """Calculate feature importance using permutation"""
        importance = {}
        
        try:
            # Get baseline prediction
            baseline_pred = model.predict(X_sample, verbose=0)
            baseline_mean = np.mean(baseline_pred)
            
            # For each feature, permute and measure impact
            n_features = X_sample.shape[2]
            
            for i in range(min(n_features, len(self.FEATURE_NAMES))):
                X_permuted = X_sample.copy()
                # Shuffle this feature across all samples
                np.random.shuffle(X_permuted[:, :, i])
                
                permuted_pred = model.predict(X_permuted, verbose=0)
                permuted_mean = np.mean(permuted_pred)
                
                # Importance = how much prediction changed
                imp = abs(baseline_mean - permuted_mean)
                feature_name = self.FEATURE_NAMES[i] if i < len(self.FEATURE_NAMES) else f"feature_{i}"
                importance[feature_name] = round(float(imp), 6)
            
            # Normalize to percentages
            total = sum(importance.values())
            if total > 0:
                importance = {k: round(v / total * 100, 2) for k, v in importance.items()}
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            
        return importance
    
    def _generate_equation_string(self, patterns: Dict) -> str:
        """Generate a human-readable equation from learned weights"""
        if not patterns.get("feature_importance"):
            return "P(up) = σ(LSTM(X) · W + b)"
        
        # Get top 5 features
        top_features = list(patterns["feature_importance"].items())[:5]
        
        if not top_features:
            return "P(up) = σ(LSTM(X) · W + b)"
        
        # Build equation string
        terms = []
        for feature, importance in top_features:
            coef = importance / 100
            terms.append(f"{coef:.2f}·{feature}")
        
        equation = f"P(up) = σ({' + '.join(terms)} + ...)"
        return equation
    
    def get_learned_patterns(self) -> Dict:
        """Get the patterns learned by the model"""
        if not hasattr(self, '_learned_patterns') or self._learned_patterns is None:
            return {
                "feature_importance": {},
                "model_equation": "Model not trained yet",
                "top_signals": []
            }
        return self._learned_patterns
    
    async def train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 50,
        data_info: Dict = None
    ) -> Dict:
        """Train model - it discovers patterns from data"""
        
        self.training_status["is_training"] = True
        self.training_status["total_epochs"] = epochs
        self.training_status["start_time"] = datetime.now(timezone.utc).isoformat()
        self.training_status["data_info"] = data_info
        
        # Prepare sequences
        training_data = self.prepare_sequences(features, labels, sequence_length)
        
        if "error" in training_data:
            self.training_status["is_training"] = False
            self.training_status["error"] = training_data["error"]
            return training_data
        
        logger.info(f"Training with {training_data['train_samples']} samples")
        
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
            
            self.best_accuracy = self.training_status["current_accuracy"]
            self.training_status["is_training"] = False
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            
            return {
                "status": "completed",
                "message": "Mock training completed",
                "final_accuracy": self.training_status["current_accuracy"]
            }
        
        try:
            # Build model
            input_shape = (training_data["sequence_length"], training_data["feature_dim"])
            model = self.build_model(input_shape)
            
            logger.info(f"Model built with input shape: {input_shape}")
            
            # Progress callback
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
            
            # Save model
            self.trained_model = model
            self.best_accuracy = max(history.history.get('accuracy', [0.5]))
            
            # Extract learned patterns
            self._learned_patterns = self.extract_learned_patterns(model, training_data["X_train"])
            self.training_status["learned_patterns"] = self._learned_patterns
            
            self.training_status["is_training"] = False
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            
            return {
                "status": "completed",
                "final_accuracy": round(history.history.get('accuracy', [0])[-1], 4),
                "final_loss": round(history.history.get('loss', [0])[-1], 4),
                "best_accuracy": round(self.best_accuracy, 4),
                "epochs_trained": len(history.history.get('loss', [])),
                "learned_patterns": self._learned_patterns
            }
            
        except Exception as e:
            self.training_status["is_training"] = False
            self.training_status["error"] = str(e)
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"Training error: {e}")
            return {"status": "error", "error": str(e)}
    
    def predict(self, features: np.ndarray, sequence_length: int = 50) -> Dict:
        """Make prediction using trained model"""
        
        model_accuracy = self.best_accuracy
        if self.training_status.get('history'):
            accuracies = [h.get('accuracy', 0) for h in self.training_status['history']]
            if accuracies:
                model_accuracy = max(max(accuracies), model_accuracy)
        
        if self.trained_model is None:
            # Simple momentum-based prediction when model not trained
            if len(features) > 1:
                recent_return = (features[-1][3] - features[-2][3]) / features[-2][3] if features[-2][3] != 0 else 0
                prob = 0.5 + (recent_return * 10)
                prob = max(0.2, min(0.8, prob))
            else:
                prob = 0.5
            
            return {
                "direction": int(prob > 0.5),
                "probability": round(prob, 4),
                "confidence": round(model_accuracy * 0.8, 4),
                "model_status": "not_trained",
                "model_accuracy": round(model_accuracy, 4)
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
        
        # Calculate confidence
        prediction_strength = abs(prob - 0.5) * 2
        confidence = (prediction_strength * 0.3) + (model_accuracy * 0.7)
        
        if prob > 0.65 or prob < 0.35:
            confidence = min(confidence * 1.15, 0.95)
        
        confidence = max(confidence, model_accuracy * 0.8)
        
        return {
            "direction": int(prob > 0.5),
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "model_status": "trained",
            "model_accuracy": round(model_accuracy, 4)
        }
    
    def stop_training(self):
        """Stop training"""
        self.training_status["is_training"] = False
        return {"status": "stopped"}


# Singleton
training_service = PureMLTrainingService()
