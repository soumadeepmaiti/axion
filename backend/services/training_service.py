"""
Training Service for the Hybrid Crypto Model
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
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TrainingService:
    """Handles model training with progress tracking"""
    
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
            "error": None
        }
        self.training_task = None
        
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
            "error": None
        }
        
    def get_status(self) -> Dict:
        """Get current training status"""
        return self.training_status.copy()
    
    def prepare_training_data(
        self,
        micro_features: np.ndarray,
        macro_features: np.ndarray,
        labels: np.ndarray,
        sequence_length_micro: int = 50,
        sequence_length_macro: int = 24,
        sentiment_features: Optional[np.ndarray] = None
    ) -> Dict:
        """Prepare data for training with sequences"""
        
        if len(micro_features) == 0 or len(labels) == 0:
            return {"error": "No data available for training"}
        
        X_micro = []
        X_macro = []
        X_sentiment = []
        y = []
        
        # Create sequences
        for i in range(sequence_length_micro, len(micro_features) - 1):
            # Micro features sequence
            micro_seq = micro_features[i - sequence_length_micro:i]
            
            # Macro features - use last available within range
            macro_idx = min(i, len(macro_features) - 1)
            macro_start = max(0, macro_idx - sequence_length_macro)
            macro_seq = macro_features[macro_start:macro_idx]
            
            # Pad macro if needed
            if len(macro_seq) < sequence_length_macro:
                pad_size = sequence_length_macro - len(macro_seq)
                macro_seq = np.pad(macro_seq, ((pad_size, 0), (0, 0)), mode='constant')
            
            # Sentiment features (use mock if not provided)
            if sentiment_features is not None and i < len(sentiment_features):
                sent = sentiment_features[i]
            else:
                sent = np.array([0.0, 0.5, 0.3, 0.2])  # Mock sentiment
            
            X_micro.append(micro_seq)
            X_macro.append(macro_seq)
            X_sentiment.append(sent)
            
            if i < len(labels):
                y.append(labels[i])
        
        if len(y) == 0:
            return {"error": "Not enough data for sequence creation"}
            
        X_micro = np.array(X_micro)
        X_macro = np.array(X_macro)
        X_sentiment = np.array(X_sentiment)
        y = np.array(y)
        
        # Create confidence labels (based on how clear the direction is)
        y_confidence = np.abs(np.diff(np.concatenate([[0.5], y.astype(float)])))
        
        # Split data
        split_idx = int(len(y) * 0.8)
        
        return {
            "X_micro_train": X_micro[:split_idx],
            "X_macro_train": X_macro[:split_idx],
            "X_sentiment_train": X_sentiment[:split_idx],
            "y_direction_train": y[:split_idx],
            "y_confidence_train": y_confidence[:split_idx],
            "X_micro_val": X_micro[split_idx:],
            "X_macro_val": X_macro[split_idx:],
            "X_sentiment_val": X_sentiment[split_idx:],
            "y_direction_val": y[split_idx:],
            "y_confidence_val": y_confidence[split_idx:],
            "total_samples": len(y),
            "train_samples": split_idx,
            "val_samples": len(y) - split_idx
        }
    
    async def train_model(
        self,
        model,
        training_data: Dict,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List = None
    ) -> Dict:
        """Train the hybrid model"""
        
        if "error" in training_data:
            return training_data
            
        if not TF_AVAILABLE:
            # Mock training for demonstration
            self.training_status["is_training"] = True
            self.training_status["total_epochs"] = epochs
            self.training_status["start_time"] = datetime.now(timezone.utc).isoformat()
            
            for epoch in range(epochs):
                await asyncio.sleep(0.1)  # Simulate training time
                self.training_status["current_epoch"] = epoch + 1
                self.training_status["current_loss"] = round(1.0 / (epoch + 1) + np.random.random() * 0.1, 4)
                self.training_status["current_accuracy"] = round(0.5 + (epoch / epochs) * 0.3 + np.random.random() * 0.05, 4)
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
                "message": "Mock training completed (TensorFlow not available)",
                "final_accuracy": self.training_status["current_accuracy"],
                "final_loss": self.training_status["current_loss"]
            }
        
        try:
            self.training_status["is_training"] = True
            self.training_status["total_epochs"] = epochs
            self.training_status["start_time"] = datetime.now(timezone.utc).isoformat()
            
            # Build model if not built
            if model.model is None:
                model.build_model()
            
            # Custom callback for progress tracking
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, service):
                    super().__init__()
                    self.service = service
                    
                def on_epoch_end(self, epoch, logs=None):
                    self.service.training_status["current_epoch"] = epoch + 1
                    self.service.training_status["current_loss"] = round(logs.get('loss', 0), 4)
                    self.service.training_status["current_accuracy"] = round(logs.get('direction_accuracy', 0), 4)
                    self.service.training_status["history"].append({
                        "epoch": epoch + 1,
                        "loss": round(logs.get('loss', 0), 4),
                        "accuracy": round(logs.get('direction_accuracy', 0), 4),
                        "val_loss": round(logs.get('val_loss', 0), 4),
                        "val_accuracy": round(logs.get('val_direction_accuracy', 0), 4),
                        "auc": round(logs.get('direction_auc', 0), 4)
                    })
            
            training_callbacks = [ProgressCallback(self)]
            if callbacks:
                training_callbacks.extend(callbacks)
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_direction_accuracy',
                patience=10,
                restore_best_weights=True
            )
            training_callbacks.append(early_stopping)
            
            # Train
            history = model.model.fit(
                [
                    training_data["X_micro_train"],
                    training_data["X_macro_train"],
                    training_data["X_sentiment_train"]
                ],
                {
                    'direction': training_data["y_direction_train"],
                    'confidence': training_data["y_confidence_train"]
                },
                validation_data=(
                    [
                        training_data["X_micro_val"],
                        training_data["X_macro_val"],
                        training_data["X_sentiment_val"]
                    ],
                    {
                        'direction': training_data["y_direction_val"],
                        'confidence': training_data["y_confidence_val"]
                    }
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=training_callbacks,
                verbose=1
            )
            
            model.is_trained = True
            model.history = history.history
            
            self.training_status["is_training"] = False
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            
            return {
                "status": "completed",
                "final_accuracy": round(history.history.get('direction_accuracy', [0])[-1], 4),
                "final_loss": round(history.history.get('loss', [0])[-1], 4),
                "epochs_trained": len(history.history.get('loss', [])),
                "best_val_accuracy": round(max(history.history.get('val_direction_accuracy', [0])), 4)
            }
            
        except Exception as e:
            self.training_status["is_training"] = False
            self.training_status["error"] = str(e)
            self.training_status["end_time"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"Training error: {e}")
            return {"status": "error", "error": str(e)}
    
    def stop_training(self):
        """Request training to stop"""
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
            self.training_status["is_training"] = False
            return {"status": "stopped"}
        return {"status": "not_training"}


# Singleton instance
training_service = TrainingService()
