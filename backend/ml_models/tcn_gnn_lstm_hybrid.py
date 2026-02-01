"""
TCN-GNN-LSTM Hybrid Architecture for Financial Predictions
Simplified Functional API implementation for stability.

Multi-Channel Fusion approach:
- TCN: Temporal patterns (dilated convolutions)
- GNN: Cross-asset correlations (attention mechanism)
- LSTM: Global trend aggregation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, BatchNormalization,
    Dropout, Concatenate, GlobalAveragePooling1D, 
    MultiHeadAttention, LayerNormalization, Add
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2, name_prefix=''):
    """Single TCN block with dilated causal convolution and residual connection"""
    
    # Store input for residual
    residual = x
    
    # First dilated convolution
    x = Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name=f'{name_prefix}_conv1'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Dropout(dropout_rate, name=f'{name_prefix}_drop1')(x)
    
    # Second dilated convolution
    x = Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name=f'{name_prefix}_conv2'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = Dropout(dropout_rate, name=f'{name_prefix}_drop2')(x)
    
    # Residual connection
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding='same', name=f'{name_prefix}_res_conv')(residual)
    
    return Add(name=f'{name_prefix}_add')([x, residual])


def tcn_encoder(x, num_filters=64, kernel_size=3, num_blocks=4, dropout_rate=0.2, name='tcn'):
    """Full TCN encoder with multiple dilated blocks"""
    
    for i in range(num_blocks):
        dilation_rate = 2 ** i  # 1, 2, 4, 8
        x = tcn_block(x, num_filters, kernel_size, dilation_rate, dropout_rate, f'{name}_block{i}')
    
    x = Conv1D(num_filters, 1, activation='relu', name=f'{name}_final')(x)
    return x


def gnn_attention_block(x, units, num_heads=4, dropout_rate=0.2, name_prefix='gat'):
    """Graph Attention block using MultiHeadAttention"""
    
    # Self-attention (treating time steps as nodes)
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=units // num_heads,
        dropout=dropout_rate,
        name=f'{name_prefix}_attn'
    )(x, x)
    
    x = Add(name=f'{name_prefix}_add1')([x, Dropout(dropout_rate, name=f'{name_prefix}_drop1')(attn_output)])
    x = LayerNormalization(name=f'{name_prefix}_ln1')(x)
    
    # Feed-forward
    ffn = Dense(units, activation='relu', name=f'{name_prefix}_ffn')(x)
    x = Add(name=f'{name_prefix}_add2')([x, Dropout(dropout_rate, name=f'{name_prefix}_drop2')(ffn)])
    x = LayerNormalization(name=f'{name_prefix}_ln2')(x)
    
    return x


def gnn_encoder(x, units=64, num_layers=2, num_heads=4, dropout_rate=0.2, name='gnn'):
    """GNN encoder using stacked attention blocks"""
    
    # Project to target dimension if needed
    if x.shape[-1] != units:
        x = Dense(units, name=f'{name}_project')(x)
    
    for i in range(num_layers):
        x = gnn_attention_block(x, units, num_heads, dropout_rate, f'{name}_layer{i}')
    
    return Dense(units, activation='relu', name=f'{name}_output')(x)


def lstm_aggregator(x, units=128, num_layers=2, dropout_rate=0.3, bidirectional=True, name='lstm_agg'):
    """LSTM aggregator for global trend capture"""
    
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        
        if bidirectional:
            x = layers.Bidirectional(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate / 2,
                    kernel_regularizer=regularizers.l2(1e-4)
                ),
                name=f'{name}_bilstm{i}'
            )(x)
        else:
            x = LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate / 2,
                kernel_regularizer=regularizers.l2(1e-4),
                name=f'{name}_lstm{i}'
            )(x)
    
    return x


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
    use_wavelet_denoise: bool = False,
    use_sac_head: bool = False,
    learning_rate: float = 0.001
) -> Model:
    """
    Build the TCN-GNN-LSTM Hybrid model using functional API.
    
    Architecture:
    - TCN for local temporal patterns (dilated convolutions)
    - GNN for cross-asset correlations (multi-head attention)
    - LSTM for global trend aggregation (bidirectional)
    """
    
    # Input
    inputs = Input(shape=input_shape, name='price_input')
    
    # Parallel encoders
    # Channel A: TCN for temporal patterns
    tcn_out = tcn_encoder(
        inputs, 
        num_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        num_blocks=tcn_blocks,
        dropout_rate=dropout_rate,
        name='tcn'
    )
    
    # Channel B: GNN for cross-asset patterns
    gnn_out = gnn_encoder(
        inputs,
        units=gnn_units,
        num_layers=gnn_layers,
        num_heads=4,
        dropout_rate=dropout_rate,
        name='gnn'
    )
    
    # Fusion: Concatenate TCN and GNN outputs
    fused = Concatenate(name='feature_fusion')([tcn_out, gnn_out])
    
    # LSTM Aggregator for global trends
    aggregated = lstm_aggregator(
        fused,
        units=lstm_units,
        num_layers=lstm_layers,
        dropout_rate=dropout_rate,
        bidirectional=True,
        name='lstm_agg'
    )
    
    # Output head
    x = Dense(256, activation='relu', name='fc1')(aggregated)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Dropout(dropout_rate, name='fc1_drop')(x)
    
    x = Dense(128, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Dropout(dropout_rate, name='fc2_drop')(x)
    
    # Prediction
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


def build_hybrid_model(input_shape: Tuple[int, int], config: Dict = None) -> Model:
    """Factory function to build TCN-GNN-LSTM Hybrid model"""
    config = config or {}
    
    # Extract config values with proper type handling
    def get_int(key, default):
        val = config.get(key, default)
        if isinstance(val, (list, tuple)):
            return int(val[0]) if val else default
        return int(val) if val else default
    
    def get_float(key, default):
        val = config.get(key, default)
        if isinstance(val, (list, tuple)):
            return float(val[0]) if val else default
        return float(val) if val else default
    
    return build_tcn_gnn_lstm_hybrid(
        input_shape=input_shape,
        tcn_filters=get_int('tcn_filters', 64),
        tcn_kernel_size=get_int('tcn_kernel_size', 3),
        tcn_blocks=get_int('tcn_blocks', 4),
        gnn_units=get_int('gnn_units', 64),
        gnn_layers=get_int('gnn_layers', 2),
        lstm_units=get_int('lstm_units', 128),
        lstm_layers=get_int('lstm_layers', 2),
        dropout_rate=get_float('dropout_rate', 0.3),
        use_wavelet_denoise=config.get('use_wavelet_denoise', False),
        use_sac_head=config.get('use_sac_head', False),
        learning_rate=get_float('learning_rate', 0.001)
    )
