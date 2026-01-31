"""
Multi-Model Training System

Enables training multiple models simultaneously and combining their predictions
for improved accuracy. Implements several ensemble strategies:

1. **Simple Voting** - Each model votes, majority wins
2. **Weighted Voting** - Models weighted by their validation accuracy
3. **Stacking** - Meta-learner trained on model outputs
4. **Blending** - Linear combination optimized on holdout set

Usage:
- Train multiple model types (LSTM, GRU, Transformer, etc.)
- Combine predictions using various ensemble methods
- Get higher accuracy than any single model
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging
import joblib
import os
from datetime import datetime
from pathlib import Path

from ml_models.advanced_models import (
    build_lstm_model, build_gru_model, build_transformer_model,
    build_cnn_lstm_model, ProgressCallback
)

logger = logging.getLogger(__name__)


class MultiModelEnsemble:
    """
    Multi-Model Ensemble System
    
    Train multiple different architectures and combine their predictions
    for improved accuracy and robustness.
    """
    
    SUPPORTED_MODELS = ['lstm', 'gru', 'transformer', 'cnn_lstm']
    ENSEMBLE_METHODS = ['voting', 'weighted', 'stacking', 'blending']
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict = None):
        self.input_shape = input_shape
        self.config = config or {}
        
        self.models: Dict[str, Model] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict] = {}
        
        # Meta-learner for stacking
        self.meta_learner = None
        
        # Blend weights for blending
        self.blend_weights: Optional[np.ndarray] = None
        
        self.is_trained = False
        self.ensemble_method = self.config.get('ensemble_method', 'weighted')
        self.selected_models = self.config.get('models', ['lstm', 'gru', 'transformer'])
    
    def build_models(self):
        """Build all selected model architectures"""
        model_builders = {
            'lstm': lambda: build_lstm_model(
                self.input_shape,
                num_lstm_layers=self.config.get('num_lstm_layers', 2),
                lstm_units=self.config.get('lstm_units', [128, 64]),
                dropout_rate=self.config.get('dropout_rate', 0.3),
                use_attention=self.config.get('use_attention', True),
                learning_rate=self.config.get('learning_rate', 0.001)
            ),
            'gru': lambda: build_gru_model(
                self.input_shape,
                num_gru_layers=self.config.get('num_lstm_layers', 2),
                gru_units=self.config.get('lstm_units', [128, 64]),
                dropout_rate=self.config.get('dropout_rate', 0.3),
                use_attention=self.config.get('use_attention', True),
                learning_rate=self.config.get('learning_rate', 0.001)
            ),
            'transformer': lambda: build_transformer_model(
                self.input_shape,
                num_transformer_blocks=self.config.get('num_lstm_layers', 2),
                dropout_rate=self.config.get('dropout_rate', 0.3),
                learning_rate=self.config.get('learning_rate', 0.001)
            ),
            'cnn_lstm': lambda: build_cnn_lstm_model(
                self.input_shape,
                dropout_rate=self.config.get('dropout_rate', 0.3),
                learning_rate=self.config.get('learning_rate', 0.001)
            )
        }
        
        for model_type in self.selected_models:
            if model_type in model_builders:
                logger.info(f"Building {model_type.upper()} model...")
                self.models[model_type] = model_builders[model_type]()
                self.model_weights[model_type] = 1.0 / len(self.selected_models)
        
        logger.info(f"Built {len(self.models)} models: {list(self.models.keys())}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 32,
            status_callback=None) -> Dict:
        """
        Train all models in the ensemble
        
        Each model is trained independently, then combined using the
        selected ensemble method.
        """
        if not self.models:
            self.build_models()
        
        training_results = {}
        all_val_predictions = []
        
        for i, (model_type, model) in enumerate(self.models.items()):
            logger.info(f"Training {model_type.upper()} model ({i+1}/{len(self.models)})...")
            
            if status_callback:
                status_callback({
                    'current_model': model_type,
                    'model_index': i + 1,
                    'total_models': len(self.models),
                    'phase': 'training'
                })
            
            # Train model
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping_patience', 10),
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
            
            # Store metrics
            self.model_metrics[model_type] = {
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc),
                'val_auc': float(val_auc),
                'epochs_trained': len(history.history['loss'])
            }
            
            training_results[model_type] = self.model_metrics[model_type]
            
            # Get validation predictions for ensemble training
            val_pred = model.predict(X_val, verbose=0).flatten()
            all_val_predictions.append(val_pred)
            
            logger.info(f"  {model_type.upper()} - Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Convert predictions to array
        val_predictions = np.column_stack(all_val_predictions)
        
        # Calculate model weights based on accuracy
        self._calculate_weights()
        
        # Train meta-learner if using stacking
        if self.ensemble_method == 'stacking':
            self._train_meta_learner(val_predictions, y_val)
        
        # Optimize blend weights if using blending
        elif self.ensemble_method == 'blending':
            self._optimize_blend_weights(val_predictions, y_val)
        
        self.is_trained = True
        
        # Calculate ensemble accuracy
        ensemble_pred = self.predict(X_val)
        ensemble_acc = float(np.mean((ensemble_pred > 0.5) == y_val))
        
        return {
            'models': training_results,
            'ensemble_accuracy': ensemble_acc,
            'model_weights': self.model_weights,
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.models)
        }
    
    def _calculate_weights(self):
        """Calculate model weights based on validation accuracy"""
        if self.ensemble_method in ['weighted', 'voting']:
            # Weight by accuracy
            accuracies = {m: self.model_metrics[m]['val_accuracy'] for m in self.models}
            total = sum(accuracies.values())
            
            for model_type in self.models:
                self.model_weights[model_type] = accuracies[model_type] / total
            
            logger.info(f"Model weights (by accuracy): {self.model_weights}")
    
    def _train_meta_learner(self, predictions: np.ndarray, y_true: np.ndarray):
        """Train meta-learner for stacking ensemble"""
        logger.info("Training meta-learner (stacking)...")
        
        # Use XGBoost as meta-learner
        self.meta_learner = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        
        self.meta_learner.fit(predictions, y_true)
        
        # Evaluate meta-learner
        meta_pred = self.meta_learner.predict(predictions)
        meta_acc = np.mean(meta_pred == y_true)
        logger.info(f"Meta-learner accuracy: {meta_acc:.4f}")
    
    def _optimize_blend_weights(self, predictions: np.ndarray, y_true: np.ndarray):
        """Optimize blend weights using simple grid search"""
        logger.info("Optimizing blend weights...")
        
        best_acc = 0
        best_weights = None
        num_models = predictions.shape[1]
        
        # Grid search over weights
        for i in range(11):
            for j in range(11 - i):
                k = 10 - i - j
                if num_models >= 3:
                    weights = np.array([i/10, j/10, k/10] + [0] * (num_models - 3))
                else:
                    weights = np.array([i/10, j/10][:num_models])
                    weights = weights / weights.sum()
                
                # Calculate blended prediction
                blended = np.sum(predictions * weights, axis=1)
                acc = np.mean((blended > 0.5) == y_true)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
        
        self.blend_weights = best_weights
        logger.info(f"Best blend weights: {self.blend_weights} (acc: {best_acc:.4f})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble prediction
        
        Combines predictions from all models using the selected method.
        """
        if not self.is_trained:
            return np.array([0.5] * len(X))
        
        # Get predictions from all models
        predictions = []
        for model_type in self.models:
            pred = self.models[model_type].predict(X, verbose=0).flatten()
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Combine using selected method
        if self.ensemble_method == 'voting':
            # Simple voting (binary)
            binary_preds = (predictions > 0.5).astype(int)
            ensemble_pred = np.mean(binary_preds, axis=1)
            
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = np.array([self.model_weights[m] for m in self.models])
            ensemble_pred = np.average(predictions, weights=weights, axis=1)
            
        elif self.ensemble_method == 'stacking':
            # Use meta-learner
            if self.meta_learner is not None:
                ensemble_pred = self.meta_learner.predict_proba(predictions)[:, 1]
            else:
                ensemble_pred = np.mean(predictions, axis=1)
                
        elif self.ensemble_method == 'blending':
            # Use optimized blend weights
            if self.blend_weights is not None:
                ensemble_pred = np.sum(predictions * self.blend_weights, axis=1)
            else:
                ensemble_pred = np.mean(predictions, axis=1)
        else:
            ensemble_pred = np.mean(predictions, axis=1)
        
        return ensemble_pred
    
    def predict_with_breakdown(self, X: np.ndarray) -> Dict:
        """
        Get ensemble prediction with individual model breakdown
        
        Useful for understanding which models agree/disagree.
        """
        if not self.is_trained:
            return {'ensemble': np.array([0.5] * len(X)), 'individual': {}}
        
        individual = {}
        for model_type in self.models:
            pred = self.models[model_type].predict(X, verbose=0).flatten()
            individual[model_type] = {
                'prediction': pred,
                'direction': 'LONG' if pred[0] > 0.5 else 'SHORT',
                'confidence': abs(pred[0] - 0.5) * 2,
                'weight': self.model_weights[model_type]
            }
        
        ensemble_pred = self.predict(X)
        
        # Calculate agreement score
        directions = [1 if individual[m]['prediction'][0] > 0.5 else 0 for m in individual]
        agreement = max(directions.count(1), directions.count(0)) / len(directions)
        
        return {
            'ensemble': ensemble_pred,
            'ensemble_direction': 'LONG' if ensemble_pred[0] > 0.5 else 'SHORT',
            'ensemble_confidence': abs(ensemble_pred[0] - 0.5) * 2,
            'agreement_score': agreement,
            'individual': individual,
            'method': self.ensemble_method
        }
    
    def save(self, path: str):
        """Save all models and ensemble configuration"""
        os.makedirs(path, exist_ok=True)
        
        # Save individual models
        for model_type, model in self.models.items():
            model.save(os.path.join(path, f'{model_type}_model.keras'))
        
        # Save meta-learner if exists
        if self.meta_learner is not None:
            self.meta_learner.save_model(os.path.join(path, 'meta_learner.json'))
        
        # Save config and weights
        config_data = {
            'config': self.config,
            'model_weights': self.model_weights,
            'model_metrics': self.model_metrics,
            'blend_weights': self.blend_weights.tolist() if self.blend_weights is not None else None,
            'ensemble_method': self.ensemble_method,
            'selected_models': self.selected_models,
            'input_shape': self.input_shape
        }
        joblib.dump(config_data, os.path.join(path, 'ensemble_config.joblib'))
        
        logger.info(f"Multi-Model Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MultiModelEnsemble':
        """Load ensemble from disk"""
        config_data = joblib.load(os.path.join(path, 'ensemble_config.joblib'))
        
        ensemble = cls(
            input_shape=tuple(config_data['input_shape']),
            config=config_data['config']
        )
        
        ensemble.model_weights = config_data['model_weights']
        ensemble.model_metrics = config_data['model_metrics']
        ensemble.ensemble_method = config_data['ensemble_method']
        ensemble.selected_models = config_data['selected_models']
        
        if config_data['blend_weights'] is not None:
            ensemble.blend_weights = np.array(config_data['blend_weights'])
        
        # Load individual models
        for model_type in ensemble.selected_models:
            model_path = os.path.join(path, f'{model_type}_model.keras')
            if os.path.exists(model_path):
                ensemble.models[model_type] = keras.models.load_model(model_path)
        
        # Load meta-learner if exists
        meta_path = os.path.join(path, 'meta_learner.json')
        if os.path.exists(meta_path):
            ensemble.meta_learner = xgb.XGBClassifier()
            ensemble.meta_learner.load_model(meta_path)
        
        ensemble.is_trained = True
        
        logger.info(f"Multi-Model Ensemble loaded from {path}")
        return ensemble


class ModelComparison:
    """
    Utility class for comparing multiple models on the same dataset
    """
    
    @staticmethod
    def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      models: List[str] = None,
                      epochs: int = 30, batch_size: int = 32) -> Dict:
        """
        Train and compare multiple models
        
        Returns comparison metrics for each model type.
        """
        models = models or ['lstm', 'gru', 'transformer', 'cnn_lstm']
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        results = {}
        
        for model_type in models:
            logger.info(f"Training {model_type.upper()} for comparison...")
            
            ensemble = MultiModelEnsemble(input_shape, {'models': [model_type]})
            ensemble.build_models()
            
            # Train single model
            model = ensemble.models[model_type]
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ],
                verbose=0
            )
            
            val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
            
            results[model_type] = {
                'val_accuracy': float(val_acc),
                'val_auc': float(val_auc),
                'val_loss': float(val_loss),
                'epochs_trained': len(history.history['loss']),
                'final_train_acc': float(history.history['accuracy'][-1]),
                'final_train_loss': float(history.history['loss'][-1])
            }
            
            # Clean up
            del model
            keras.backend.clear_session()
        
        # Rank models
        ranked = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
        
        return {
            'results': results,
            'ranking': [{'rank': i+1, 'model': m, 'accuracy': r['val_accuracy']} 
                       for i, (m, r) in enumerate(ranked)],
            'best_model': ranked[0][0],
            'best_accuracy': ranked[0][1]['val_accuracy']
        }
