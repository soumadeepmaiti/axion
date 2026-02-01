"""
Advanced Training Service with:
- Walk-Forward Validation
- Class Balancing
- Optuna Hyperparameter Search
- Multiple Network Architectures
- Reinforcement Learning (DQN, PPO)
- Multi-Model Ensemble Training
- Model Persistence
- Learning Rate Scheduling
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from typing import Dict, List, Optional, Tuple
import optuna
from optuna.samplers import TPESampler
import threading
import logging
from datetime import datetime, timezone
import asyncio

from ml_models.advanced_models import (
    build_model, EnsembleModel, ProgressCallback,
    save_model, load_model, list_saved_models, MODEL_DIR,
    build_rl_agent, get_multi_model_ensemble
)
from services.advanced_data_pipeline import advanced_data_pipeline

logger = logging.getLogger(__name__)


class AdvancedTrainingService:
    """
    Comprehensive training service with advanced features
    """
    
    def __init__(self):
        self.model = None
        self.ensemble_model = None
        self.is_training = False
        self.training_thread = None
        self.stop_requested = False
        
        self.status = {
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0,
            'current_accuracy': 0,
            'history': [],
            'start_time': None,
            'mode': None,
            'network_type': None,
            'data_info': None,
            'learned_patterns': None,
            'validation_results': None,
            'error': None
        }
        
        self.best_model = None
        self.best_accuracy = 0
        self.feature_names = []
        self.scaler = None
    
    def get_status(self) -> Dict:
        """Get current training status"""
        return self.status.copy()
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray, 
                        sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def apply_class_balancing(self, X: np.ndarray, y: np.ndarray, 
                             method: str = 'class_weight') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply class balancing techniques"""
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        if method == 'smote':
            try:
                # Flatten for SMOTE
                X_flat = X.reshape(X.shape[0], -1)
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_flat, y)
                # Reshape back
                X = X_resampled.reshape(-1, X.shape[1], X.shape[2])
                y = y_resampled
                logger.info(f"Applied SMOTE: {len(y)} samples")
            except Exception as e:
                logger.warning(f"SMOTE failed, using class weights: {e}")
        
        return X, y, class_weight_dict
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, 
                               n_splits: int = 5,
                               network_type: str = 'lstm',
                               config: Dict = None) -> Dict:
        """
        Walk-forward (time series) cross-validation
        Returns validation metrics across folds
        """
        config = config or {}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Walk-forward fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build fresh model for each fold
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_model(network_type, input_shape, config)
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Fewer epochs for CV
                batch_size=config.get('batch_size', 32),
                verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Evaluate
            val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
            
            fold_results.append({
                'fold': fold + 1,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc),
                'val_auc': float(val_auc)
            })
            
            # Clean up
            del model
            keras.backend.clear_session()
        
        # Aggregate results
        avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
        std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
        
        return {
            'folds': fold_results,
            'avg_accuracy': float(avg_accuracy),
            'std_accuracy': float(std_accuracy),
            'n_splits': n_splits
        }
    
    def optuna_hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray,
                                    network_type: str = 'lstm',
                                    n_trials: int = 20) -> Dict:
        """
        Automated hyperparameter search using Optuna
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        def objective(trial):
            # Hyperparameters to tune
            config = {
                'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 3),
                'lstm_units': [
                    trial.suggest_categorical('lstm_units_0', [32, 64, 128, 256]),
                    trial.suggest_categorical('lstm_units_1', [32, 64, 128]),
                    trial.suggest_categorical('lstm_units_2', [32, 64]),
                ],
                'num_dense_layers': trial.suggest_int('num_dense_layers', 1, 3),
                'dense_units': [
                    trial.suggest_categorical('dense_units_0', [32, 64, 128]),
                    trial.suggest_categorical('dense_units_1', [16, 32, 64]),
                    trial.suggest_categorical('dense_units_2', [16, 32]),
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'use_attention': trial.suggest_categorical('use_attention', [True, False]),
                'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            }
            
            model = build_model(network_type, input_shape, config)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            val_loss, val_acc, _ = model.evaluate(X_val, y_val, verbose=0)
            
            # Clean up
            del model
            keras.backend.clear_session()
            
            return val_acc
        
        # Run optimization
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_accuracy': float(study.best_value),
            'n_trials': n_trials,
            'trials': [
                {'number': t.number, 'value': t.value, 'params': t.params}
                for t in study.trials[:10]  # Top 10
            ]
        }
    
    def get_lr_schedule(self, initial_lr: float = 0.001, 
                       schedule_type: str = 'cosine') -> LearningRateScheduler:
        """Create learning rate schedule"""
        
        if schedule_type == 'cosine':
            def cosine_decay(epoch, lr):
                total_epochs = self.status['total_epochs']
                return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
            return LearningRateScheduler(cosine_decay)
        
        elif schedule_type == 'step':
            def step_decay(epoch, lr):
                drop_rate = 0.5
                epochs_drop = 20
                return initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
            return LearningRateScheduler(step_decay)
        
        else:
            return ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
    
    def _run_training(self, data: Dict, config: Dict):
        """Internal training loop (runs in thread)"""
        try:
            self.status['is_training'] = True
            self.status['error'] = None
            self.status['history'] = []
            
            features = data['features']
            labels = data['labels']
            prices = data.get('prices', features[:, 3] if features.shape[1] > 3 else features[:, 0])  # Close prices
            self.feature_names = data.get('feature_names', [])
            
            sequence_length = config.get('sequence_length', 50)
            network_type = config.get('network_type', 'lstm')
            use_ensemble = network_type == 'ensemble'
            use_rl = network_type.startswith('rl_')
            use_multi_model = network_type == 'multi_model'
            
            self.status['network_type'] = network_type
            
            # Create sequences
            X, y = self.create_sequences(features, labels, sequence_length)
            
            # Also create price sequences for RL
            if use_rl:
                _, price_seq = self.create_sequences(prices.reshape(-1, 1), prices[sequence_length:], sequence_length)
                price_seq = prices[sequence_length:len(X) + sequence_length]
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data: {len(X)} sequences (need at least 100)")
            
            # Train/validation split (time-based)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            if use_rl:
                prices_train = prices[sequence_length:sequence_length + split_idx]
                prices_val = prices[sequence_length + split_idx:sequence_length + len(X)]
            
            # Apply class balancing (for non-RL models)
            if not use_rl:
                balance_method = config.get('class_balance_method', 'class_weight')
                X_train, y_train, class_weights = self.apply_class_balancing(
                    X_train, y_train, balance_method
                )
            else:
                class_weights = None
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Walk-forward validation (optional, not for RL)
            if config.get('use_walk_forward', False) and not use_rl:
                logger.info("Running walk-forward validation...")
                wf_results = self.walk_forward_validation(
                    X_train, y_train, n_splits=config.get('cv_folds', 5),
                    network_type=network_type, config=config
                )
                self.status['validation_results'] = wf_results
                logger.info(f"Walk-forward CV accuracy: {wf_results['avg_accuracy']:.4f} ± {wf_results['std_accuracy']:.4f}")
            
            # Hyperparameter search (optional, not for RL)
            if config.get('use_optuna', False) and not use_rl:
                logger.info("Running Optuna hyperparameter search...")
                optuna_results = self.optuna_hyperparameter_search(
                    X_train, y_train, X_val, y_val,
                    network_type=network_type,
                    n_trials=config.get('optuna_trials', 20)
                )
                config.update(optuna_results['best_params'])
                self.status['optuna_results'] = optuna_results
                logger.info(f"Best Optuna params: {optuna_results['best_params']}")
            
            # ========================
            # Reinforcement Learning
            # ========================
            if use_rl:
                rl_algorithm = 'dqn' if network_type == 'rl_dqn' else 'ppo'
                logger.info(f"Training {rl_algorithm.upper()} Reinforcement Learning Agent...")
                
                from ml_models.rl_models import RLTrainer
                
                rl_config = {
                    'gamma': config.get('rl_gamma', 0.99),
                    'epsilon_decay': 0.995,
                    'batch_size': config.get('batch_size', 64)
                }
                
                self.rl_trainer = RLTrainer(algorithm=rl_algorithm, config=rl_config)
                self.rl_trainer.setup(X_train.reshape(len(X_train), -1), prices_train, window_size=sequence_length)
                
                num_episodes = config.get('rl_episodes', 100)
                self.status['total_epochs'] = num_episodes
                
                def rl_callback(status):
                    self.status['current_epoch'] = status['episode']
                    self.status['progress'] = (status['episode'] / status['total_episodes']) * 100
                    self.status['history'].append({
                        'epoch': status['episode'],
                        'reward': status['avg_reward'],
                        'pnl': status['avg_pnl']
                    })
                
                rl_results = self.rl_trainer.train(num_episodes=num_episodes, status_callback=rl_callback)
                
                final_accuracy = (rl_results['final_avg_pnl'] > 0)  # Profitable = success
                self.status['final_accuracy'] = 1.0 if final_accuracy else 0.0
                self.status['rl_results'] = rl_results
                self.status['learned_patterns'] = {
                    'algorithm': rl_algorithm.upper(),
                    'final_avg_pnl': rl_results['final_avg_pnl'],
                    'best_pnl': rl_results['best_pnl'],
                    'total_episodes': rl_results['total_episodes']
                }
                
                # Save RL model
                if config.get('save_model', True):
                    model_path = str(MODEL_DIR / f"{data['data_info']['symbol'].replace('/', '_')}_{rl_algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    import os
                    os.makedirs(model_path, exist_ok=True)
                    self.rl_trainer.save(model_path)
                    self.status['saved_model_path'] = model_path
                
                logger.info(f"RL Training completed. Final avg PnL: ${rl_results['final_avg_pnl']:.2f}")
                
            # ========================
            # Multi-Model Ensemble
            # ========================
            elif use_multi_model:
                logger.info("Training Multi-Model Ensemble...")
                
                selected_models = config.get('selected_models', ['lstm', 'gru', 'transformer'])
                ensemble_method = config.get('ensemble_method', 'weighted')
                
                ensemble_config = {
                    **config,
                    'models': selected_models,
                    'ensemble_method': ensemble_method
                }
                
                self.multi_model_ensemble = get_multi_model_ensemble(input_shape, ensemble_config)
                
                def mm_callback(status):
                    model_progress = (status['model_index'] / status['total_models']) * 100
                    self.status['progress'] = model_progress
                    self.status['current_model'] = status['current_model']
                
                results = self.multi_model_ensemble.fit(
                    X_train, y_train, X_val, y_val,
                    epochs=self.status['total_epochs'],
                    batch_size=config.get('batch_size', 32),
                    status_callback=mm_callback
                )
                
                self.status['final_accuracy'] = results['ensemble_accuracy']
                self.status['multi_model_results'] = results
                self.status['learned_patterns'] = {
                    'ensemble_method': ensemble_method,
                    'num_models': results['num_models'],
                    'model_weights': results['model_weights'],
                    'individual_accuracies': {m: results['models'][m]['val_accuracy'] for m in results['models']}
                }
                
                # Save multi-model ensemble
                if config.get('save_model', True):
                    model_path = str(MODEL_DIR / f"{data['data_info']['symbol'].replace('/', '_')}_multi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    self.multi_model_ensemble.save(model_path)
                    self.status['saved_model_path'] = model_path
                
                logger.info(f"Multi-Model Training completed. Ensemble accuracy: {results['ensemble_accuracy']:.4f}")
            
            # ========================
            # Standard Ensemble
            # ========================
            elif use_ensemble:
                logger.info("Building Ensemble model (LSTM + XGBoost + RandomForest)...")
                self.ensemble_model = EnsembleModel(input_shape, config)
                
                history = self.ensemble_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.status['total_epochs'],
                    batch_size=config.get('batch_size', 32),
                    callbacks=[ProgressCallback(self.status)]
                )
                
                self.model = self.ensemble_model.nn_model
                
                y_pred_proba = self.ensemble_model.predict(X_val)
                y_pred = (y_pred_proba > 0.5).astype(int)
                final_accuracy = float(np.mean(y_pred == y_val))
                
                self.status['final_accuracy'] = final_accuracy
                self.status['learned_patterns'] = self._generate_learned_patterns(
                    self.model, X_val, y_val, self.feature_names
                )
                
                # Save model if requested
                if config.get('save_model', True):
                    metrics = {
                        'final_accuracy': final_accuracy,
                        'final_loss': float(self.status['history'][-1]['val_loss']) if self.status['history'] else 0,
                        'epochs_trained': self.status['current_epoch']
                    }
                    model_path = str(MODEL_DIR / f"{data['data_info']['symbol'].replace('/', '_')}_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    self.ensemble_model.save(model_path)
                    self.status['saved_model_path'] = model_path
                
                logger.info(f"Ensemble Training completed. Final accuracy: {final_accuracy:.4f}")
                
            # ========================
            # Standard Neural Network
            # ========================
            else:
                logger.info(f"Building {network_type.upper()} model...")
                self.model = build_model(network_type, input_shape, config)
                
                # Callbacks
                callbacks = [
                    ProgressCallback(self.status),
                    self.get_lr_schedule(
                        config.get('learning_rate', 0.001),
                        config.get('lr_schedule', 'reduce_plateau')
                    )
                ]
                
                # Optional early stopping
                if config.get('use_early_stopping', True):
                    callbacks.append(
                        EarlyStopping(
                            monitor='val_loss',
                            patience=config.get('early_stopping_patience', 15),
                            restore_best_weights=True
                        )
                    )
                
                # Train
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.status['total_epochs'],
                    batch_size=config.get('batch_size', 32),
                    class_weight=class_weights if class_weights and balance_method == 'class_weight' else None,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                val_loss, val_acc, val_auc = self.model.evaluate(X_val, y_val, verbose=0)
                final_accuracy = float(val_acc)
                
                # Save model if requested
                if config.get('save_model', True):
                    metrics = {
                        'final_accuracy': final_accuracy,
                        'final_loss': float(self.status['history'][-1]['val_loss']) if self.status['history'] else 0,
                        'epochs_trained': self.status['current_epoch']
                    }
                    model_path = save_model(
                        self.model, 
                        data['data_info']['symbol'],
                        network_type,
                        metrics,
                        config
                    )
                    self.status['saved_model_path'] = model_path
                
                # Generate learned patterns summary
                self.status['learned_patterns'] = self._generate_learned_patterns(
                    self.model, X_val, y_val, self.feature_names
                )
                
                self.status['final_accuracy'] = final_accuracy
                logger.info(f"Training completed. Final accuracy: {final_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            self.status['error'] = str(e)
        finally:
            self.status['is_training'] = False
            self.is_training = False
    
    def _generate_learned_patterns(self, model, X_val, y_val, feature_names) -> Dict:
        """Generate interpretable learned patterns"""
        try:
            # Get predictions
            y_pred = model.predict(X_val, verbose=0).flatten()
            
            # Feature importance (simplified - based on input weights)
            importance = {}
            if feature_names:
                # Use last timestep variance as proxy for importance
                for i, name in enumerate(feature_names[:20]):
                    if i < X_val.shape[2]:
                        importance[name] = float(np.std(X_val[:, :, i]) * 100)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
            
            # Generate equation string
            top_features = list(sorted_importance.items())[:6]
            equation_parts = [f"{v/100:.2f}·{k}" for k, v in top_features]
            equation = f"P(up) = σ({' + '.join(equation_parts)} + ...)"
            
            return {
                'feature_importance': sorted_importance,
                'model_equation': equation,
                'prediction_mean': float(np.mean(y_pred)),
                'prediction_std': float(np.std(y_pred))
            }
        except Exception as e:
            logger.warning(f"Could not generate learned patterns: {e}")
            return {}
    
    async def start_training(self, symbol: str, config: Dict) -> Dict:
        """Start training with comprehensive configuration"""
        
        if self.is_training:
            return {'status': 'error', 'message': 'Training already in progress'}
        
        self.stop_requested = False
        self.is_training = True
        
        # Reset status
        self.status = {
            'is_training': True,
            'current_epoch': 0,
            'total_epochs': config.get('epochs', 100),
            'current_loss': 0,
            'current_accuracy': 0,
            'history': [],
            'start_time': datetime.now(timezone.utc).isoformat(),
            'mode': config.get('mode', 'pure_ml'),
            'network_type': config.get('network_type', 'lstm'),
            'data_info': None,
            'learned_patterns': None,
            'validation_results': None,
            'error': None
        }
        
        # Prepare data
        try:
            # Use real_data_only=True by default (no mocked on-chain, sentiment)
            real_data_only = config.get('real_data_only', True)
            
            data = await advanced_data_pipeline.prepare_training_data(
                symbol=symbol,
                start_date=config.get('start_date'),
                end_date=config.get('end_date'),
                timeframe=config.get('timeframe', '1h'),
                multi_timeframe=config.get('multi_timeframe', False),
                real_data_only=real_data_only
            )
            
            if len(data['features']) < 100:
                self.is_training = False
                self.status['is_training'] = False
                return {
                    'status': 'error',
                    'message': f"Insufficient data: {len(data['features'])} samples"
                }
            
            self.status['data_info'] = data['data_info']
            self.status['real_data_only'] = real_data_only
            
        except Exception as e:
            self.is_training = False
            self.status['is_training'] = False
            return {'status': 'error', 'message': str(e)}
        
        # Start training in background thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(data, config)
        )
        self.training_thread.start()
        
        return {
            'status': 'started',
            'message': f"Training started with {config.get('network_type', 'lstm')} model",
            'config': {
                'network_type': config.get('network_type', 'lstm'),
                'mode': config.get('mode', 'pure_ml'),
                'epochs': config.get('epochs', 100),
                'samples': len(data['features']),
                'features': len(data.get('feature_names', [])),
                'use_walk_forward': config.get('use_walk_forward', False),
                'use_optuna': config.get('use_optuna', False)
            }
        }
    
    def stop_training(self) -> Dict:
        """Stop current training"""
        if not self.is_training:
            return {'status': 'not_training', 'message': 'No training in progress'}
        
        self.stop_requested = True
        self.status['is_training'] = False
        self.is_training = False
        
        return {'status': 'stopped', 'message': 'Training stop requested'}
    
    def predict(self, features: np.ndarray, sequence_length: int = 50) -> Dict:
        """Make prediction using trained model"""
        
        if self.model is None and self.ensemble_model is None:
            return {
                'direction': 0,
                'probability': 0.5,
                'confidence': 0.5,
                'model_status': 'not_trained'
            }
        
        try:
            # Prepare sequence
            if len(features) >= sequence_length:
                X = features[-sequence_length:].reshape(1, sequence_length, -1)
            else:
                # Pad if necessary
                padding = np.zeros((sequence_length - len(features), features.shape[1]))
                padded = np.vstack([padding, features])
                X = padded.reshape(1, sequence_length, -1)
            
            # Predict
            if self.ensemble_model is not None:
                prob = self.ensemble_model.predict(X)[0]
            else:
                prob = float(self.model.predict(X, verbose=0)[0][0])
            
            direction = 1 if prob > 0.5 else 0
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            
            # Factor in model accuracy
            model_acc = self.status.get('final_accuracy', 0.5)
            adjusted_confidence = confidence * 0.4 + model_acc * 0.6
            
            return {
                'direction': direction,
                'probability': float(prob),
                'confidence': float(adjusted_confidence),
                'model_status': 'trained',
                'model_accuracy': float(model_acc),
                'network_type': self.status.get('network_type', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'direction': 0,
                'probability': 0.5,
                'confidence': 0.5,
                'model_status': 'error',
                'error': str(e)
            }
    
    def get_saved_models(self) -> List[Dict]:
        """Get list of saved models"""
        return list_saved_models()
    
    async def load_saved_model(self, model_path: str) -> Dict:
        """Load a previously saved model"""
        try:
            self.model, metadata = load_model(model_path)
            self.status['network_type'] = metadata.get('network_type', 'unknown')
            self.status['final_accuracy'] = metadata.get('metrics', {}).get('final_accuracy', 0)
            
            return {
                'status': 'loaded',
                'model_path': model_path,
                'metadata': metadata
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# Singleton instance
advanced_training_service = AdvancedTrainingService()
