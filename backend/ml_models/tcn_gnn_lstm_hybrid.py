"""
TCN-GNN-LSTM Hybrid Architecture for Financial Predictions

A Multi-Channel Fusion approach that separates:
- Temporal patterns (TCN + LSTM)
- Relational patterns (GNN for cross-asset correlations)

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│  (OHLCV + Technical Indicators + Correlation Matrix)        │
└─────────────────────────────────────────────────────────────┘
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
┌─────────────────────┐   ┌─────────────────────┐
│   WAVELET DENOISE   │   │   GRAPH BUILDER     │
│   (Remove noise)    │   │   (Asset relations) │
└─────────────────────┘   └─────────────────────┘
          │                         │
          ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│   TCN ENCODER       │   │   GNN ENCODER       │
│   (Local patterns)  │   │   (Cross-asset)     │
└─────────────────────┘   └─────────────────────┘
          │                         │
          └────────┬────────────────┘
                   ▼
        ┌─────────────────────┐
        │   LSTM AGGREGATOR   │
        │   (Global trends)   │
        └─────────────────────┘
                   │
          ┌───────┴───────┐
          ▼               ▼
┌─────────────────┐ ┌─────────────────┐
│  PREDICTION     │ │  SAC RL HEAD    │
│  (Direction)    │ │  (Trade action) │
└─────────────────┘ └─────────────────┘
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, BatchNormalization,
    Dropout, Concatenate, GlobalAveragePooling1D, 
    MultiHeadAttention, LayerNormalization, Add, Multiply
)
from typing import Dict, List, Optional, Tuple
import logging
import pywt  # PyWavelets for wavelet denoising

logger = logging.getLogger(__name__)


# ==================== WAVELET DENOISING ====================

class WaveletDenoiseLayer(layers.Layer):
    """
    Wavelet Transform layer for denoising financial time series.
    Removes market noise while preserving important signals.
    """
    
    def __init__(self, wavelet='db4', level=2, threshold_mode='soft', **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
    
    def call(self, inputs, training=None):
        # During training, apply denoising with some probability
        if training:
            return tf.numpy_function(
                self._denoise_batch, 
                [inputs], 
                tf.float32
            )
        return inputs
    
    def _denoise_batch(self, batch):
        """Apply wavelet denoising to a batch"""
        batch = np.array(batch)
        denoised = np.zeros_like(batch)
        
        for i in range(batch.shape[0]):
            for j in range(batch.shape[2]):  # Each feature
                signal = batch[i, :, j]
                try:
                    # Wavelet decomposition
                    coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                    
                    # Calculate threshold
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
                    
                    # Apply threshold to detail coefficients
                    denoised_coeffs = [coeffs[0]]  # Keep approximation
                    for k in range(1, len(coeffs)):
                        if self.threshold_mode == 'soft':
                            denoised_coeffs.append(
                                pywt.threshold(coeffs[k], threshold, mode='soft')
                            )
                        else:
                            denoised_coeffs.append(
                                pywt.threshold(coeffs[k], threshold, mode='hard')
                            )
                    
                    # Reconstruct
                    denoised[i, :, j] = pywt.waverec(denoised_coeffs, self.wavelet)[:len(signal)]
                except:
                    denoised[i, :, j] = signal
        
        return denoised.astype(np.float32)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'wavelet': self.wavelet,
            'level': self.level,
            'threshold_mode': self.threshold_mode
        })
        return config


# ==================== TCN (Temporal Convolutional Network) ====================

class TCNBlock(layers.Layer):
    """
    Temporal Convolutional Network block with dilated causal convolutions.
    Captures long-range temporal dependencies without gradient problems.
    """
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Dilated causal convolution
        self.conv1 = Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.conv2 = Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
        # Residual connection
        self.residual_conv = Conv1D(filters, 1, padding='same')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Residual
        if inputs.shape[-1] != self.filters:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs
        
        return layers.add([x, residual])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config


class TCNEncoder(layers.Layer):
    """
    Full TCN encoder with multiple dilated blocks.
    Receptive field grows exponentially with depth.
    """
    
    def __init__(self, num_filters=64, kernel_size=3, num_blocks=4, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.blocks = []
        self.final_conv = None
    
    def build(self, input_shape):
        # Create TCN blocks with exponentially increasing dilation
        for i in range(self.num_blocks):
            dilation_rate = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(
                TCNBlock(self.num_filters, self.kernel_size, dilation_rate, self.dropout_rate)
            )
        self.final_conv = Conv1D(self.num_filters, 1, activation='relu')
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return self.final_conv(x)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_filters)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks,
            'dropout_rate': self.dropout_rate
        })
        return config


# ==================== GNN (Graph Neural Network) ====================

class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention Network (GAT) layer for learning cross-asset relationships.
    Treats assets as nodes and correlations as edges.
    """
    
    def __init__(self, units, num_heads=4, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=dropout_rate
        )
        self.ffn = Dense(units, activation='relu')
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs, adjacency_matrix=None, training=None):
        # Self-attention (or masked attention if adjacency provided)
        attn_output = self.attention(inputs, inputs, training=training)
        x = self.layernorm1(inputs + self.dropout(attn_output, training=training))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout(ffn_output, training=training))


class GNNEncoder(layers.Layer):
    """
    Graph Neural Network encoder for cross-asset correlation learning.
    """
    
    def __init__(self, units=64, num_layers=2, num_heads=4, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.gat_layers = []
        self.output_dense = None
    
    def build(self, input_shape):
        for _ in range(self.num_layers):
            self.gat_layers.append(
                GraphAttentionLayer(self.units, self.num_heads, self.dropout_rate)
            )
        self.output_dense = Dense(self.units, activation='relu')
        super().build(input_shape)
    
    def call(self, inputs, adjacency_matrix=None, training=None):
        x = inputs
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adjacency_matrix, training=training)
        return self.output_dense(x)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


# ==================== LSTM AGGREGATOR ====================

class LSTMAggregator(layers.Layer):
    """
    LSTM aggregator that combines TCN and GNN features
    to capture global market trends.
    """
    
    def __init__(self, units=128, num_layers=2, dropout_rate=0.3, bidirectional=True, **kwargs):
        super().__init__(**kwargs)
        self.lstm_layers = []
        
        for i in range(num_layers):
            lstm = LSTM(
                units,
                return_sequences=(i < num_layers - 1),
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate / 2,
                kernel_regularizer=regularizers.l2(1e-4)
            )
            if bidirectional:
                lstm = layers.Bidirectional(lstm)
            self.lstm_layers.append(lstm)
        
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units // 4)
        self.layernorm = LayerNormalization()
    
    def call(self, inputs, training=None):
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return x


# ==================== SAC (Soft Actor-Critic) HEAD ====================

class SACHead(layers.Layer):
    """
    Soft Actor-Critic head for RL-based trading decisions.
    Outputs a Gaussian distribution (mean, std) for uncertainty-aware trading.
    """
    
    def __init__(self, action_dim=3, hidden_units=[256, 128], **kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim  # BUY, HOLD, SELL
        
        # Shared network
        self.shared_layers = [
            Dense(units, activation='relu', kernel_regularizer=regularizers.l2(1e-4))
            for units in hidden_units
        ]
        
        # Actor (policy) - outputs mean and log_std
        self.mean_layer = Dense(action_dim, activation='linear')
        self.log_std_layer = Dense(action_dim, activation='linear')
        
        # Critic (value) - two Q-functions for stability
        self.q1_layer = Dense(action_dim, activation='linear')
        self.q2_layer = Dense(action_dim, activation='linear')
        
        # Value function
        self.value_layer = Dense(1, activation='linear')
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)
        
        # Policy (actor)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        
        # Q-values (critic)
        q1 = self.q1_layer(x)
        q2 = self.q2_layer(x)
        
        # Value
        value = self.value_layer(x)
        
        return {
            'mean': mean,
            'std': std,
            'log_std': log_std,
            'q1': q1,
            'q2': q2,
            'value': value
        }
    
    def sample_action(self, mean, std, deterministic=False):
        """Sample action from Gaussian policy"""
        if deterministic:
            return mean
        
        # Reparameterization trick
        noise = tf.random.normal(shape=tf.shape(mean))
        action = mean + std * noise
        
        # Apply tanh squashing
        action = tf.tanh(action)
        
        return action
    
    def get_action_probs(self, mean, std):
        """Get action probabilities using softmax on mean"""
        return tf.nn.softmax(mean)


# ==================== FULL HYBRID MODEL ====================

def build_tcn_gnn_lstm_hybrid(
    input_shape: Tuple[int, int],
    tcn_filters: int = 64,
    tcn_kernel_size: int = 3,
    tcn_blocks: int = 4,
    gnn_units: int = 64,
    gnn_layers: int = 2,
    lstm_units: int = 128,
    lstm_layers: int = 2,
    dropout_rate: float = 0.3,
    use_wavelet_denoise: bool = True,
    use_sac_head: bool = False,
    learning_rate: float = 0.001
) -> Model:
    """
    Build the TCN-GNN-LSTM Hybrid model.
    
    Architecture:
    - Wavelet denoising for noise removal
    - TCN for local temporal patterns
    - GNN for cross-asset correlations
    - LSTM for global trend aggregation
    - Optional SAC head for RL trading
    """
    
    # Input
    inputs = Input(shape=input_shape, name='price_input')
    
    x = inputs
    
    # 1. Wavelet Denoising (optional)
    if use_wavelet_denoise:
        try:
            x = WaveletDenoiseLayer(wavelet='db4', level=2, name='wavelet_denoise')(x)
        except:
            logger.warning("Wavelet denoising failed, using raw input")
    
    # 2. Parallel Encoders
    # Channel A: TCN for temporal patterns
    tcn_output = TCNEncoder(
        num_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        num_blocks=tcn_blocks,
        dropout_rate=dropout_rate,
        name='tcn_encoder'
    )(x)
    
    # Channel B: GNN for cross-asset patterns
    # Reshape for GNN (treat time steps as nodes initially)
    gnn_output = GNNEncoder(
        units=gnn_units,
        num_layers=gnn_layers,
        num_heads=4,
        dropout_rate=dropout_rate,
        name='gnn_encoder'
    )(x)
    
    # 3. Fusion: Concatenate TCN and GNN outputs
    fused = Concatenate(name='feature_fusion')([tcn_output, gnn_output])
    
    # 4. LSTM Aggregator for global trends
    aggregated = LSTMAggregator(
        units=lstm_units,
        num_layers=lstm_layers,
        dropout_rate=dropout_rate,
        bidirectional=True,
        name='lstm_aggregator'
    )(fused)
    
    # 5. Output heads
    # Dense layers
    x = Dense(256, activation='relu', name='fc1')(aggregated)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    if use_sac_head:
        # SAC head for RL trading
        sac = SACHead(action_dim=3, hidden_units=[128, 64], name='sac_head')
        sac_output = sac(x)
        
        # Use mean of policy as prediction
        action_probs = sac.get_action_probs(sac_output['mean'], sac_output['std'])
        # Convert to binary (LONG probability)
        output = Dense(1, activation='sigmoid', name='prediction')(
            Concatenate()([action_probs, sac_output['value']])
        )
    else:
        # Standard prediction head
        output = Dense(1, activation='sigmoid', name='prediction')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=output, name='TCN_GNN_LSTM_Hybrid')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info(f"Built TCN-GNN-LSTM Hybrid model: {model.count_params():,} parameters")
    
    return model


# ==================== MULTISCALE INPUT PROCESSOR ====================

class MultiscaleProcessor:
    """
    Process data from multiple timeframes for multiscale analysis.
    """
    
    @staticmethod
    def create_multiscale_features(
        data_1m: np.ndarray,
        data_1h: np.ndarray,
        data_1d: np.ndarray,
        seq_length: int = 50
    ) -> np.ndarray:
        """
        Combine features from multiple timeframes.
        
        Args:
            data_1m: 1-minute data features
            data_1h: 1-hour data features  
            data_1d: 1-day data features
            seq_length: Sequence length for each timeframe
        
        Returns:
            Combined multiscale feature array
        """
        # Resample to common length
        def resample(data, target_len):
            if len(data) < target_len:
                return np.pad(data, ((target_len - len(data), 0), (0, 0)), mode='edge')
            return data[-target_len:]
        
        # Align all timeframes
        d_1m = resample(data_1m, seq_length)
        d_1h = resample(data_1h, seq_length)
        d_1d = resample(data_1d, seq_length)
        
        # Concatenate along feature dimension
        return np.concatenate([d_1m, d_1h, d_1d], axis=-1)


# ==================== CORRELATION MATRIX BUILDER ====================

class CorrelationMatrixBuilder:
    """
    Build correlation matrices for GNN adjacency.
    """
    
    @staticmethod
    def build_rolling_correlation(
        returns: np.ndarray,
        window: int = 30
    ) -> np.ndarray:
        """
        Build rolling correlation matrix for assets.
        
        Args:
            returns: Array of shape (time, num_assets)
            window: Rolling window size
        
        Returns:
            Correlation matrix of shape (num_assets, num_assets)
        """
        if len(returns) < window:
            return np.eye(returns.shape[1])
        
        # Use last `window` observations
        recent_returns = returns[-window:]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(recent_returns.T)
        
        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Apply threshold (only keep strong correlations)
        threshold = 0.3
        corr_matrix = np.where(np.abs(corr_matrix) > threshold, corr_matrix, 0)
        
        return corr_matrix


# ==================== FACTORY FUNCTION ====================

def build_hybrid_model(
    input_shape: Tuple[int, int],
    config: Dict = None
) -> Model:
    """
    Factory function to build TCN-GNN-LSTM Hybrid model.
    
    Args:
        input_shape: (sequence_length, num_features)
        config: Model configuration dict
    
    Returns:
        Compiled Keras model
    """
    config = config or {}
    
    return build_tcn_gnn_lstm_hybrid(
        input_shape=input_shape,
        tcn_filters=config.get('tcn_filters', 64),
        tcn_kernel_size=config.get('tcn_kernel_size', 3),
        tcn_blocks=config.get('tcn_blocks', 4),
        gnn_units=config.get('gnn_units', 64),
        gnn_layers=config.get('gnn_layers', 2),
        lstm_units=config.get('lstm_units', 128),
        lstm_layers=config.get('lstm_layers', 2),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_wavelet_denoise=config.get('use_wavelet_denoise', True),
        use_sac_head=config.get('use_sac_head', False),
        learning_rate=config.get('learning_rate', 0.001)
    )
