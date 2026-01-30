"""
State-of-the-Art Model Architectures
- Temporal Fusion Transformer (TFT)
- Multi-Task Learning
- Graph Neural Network for Asset Relationships
- Attention on Different Timeframes
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(layers.Layer):
    """Positional encoding for Transformer models"""
    
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(self.max_len, self.d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False
        )
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) - Core building block of TFT
    """
    
    def __init__(self, hidden_size: int, output_size: int = None, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_size, activation='elu')
        self.dense2 = layers.Dense(self.hidden_size)
        self.gate = layers.Dense(self.output_size, activation='sigmoid')
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        
        # Skip connection projection if sizes differ
        input_dim = input_shape[-1]
        if input_dim != self.output_size:
            self.skip_proj = layers.Dense(self.output_size)
        else:
            self.skip_proj = None
            
    def call(self, x, training=None):
        # Primary path
        hidden = self.dense1(x)
        hidden = self.dropout(hidden, training=training)
        hidden = self.dense2(hidden)
        
        # Gating
        gate = self.gate(x)
        gated = gate * hidden
        
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
            
        # Add & Norm
        output = self.layer_norm(skip + self.dropout(gated, training=training))
        return output


class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network (VSN) - Selects important features
    """
    
    def __init__(self, num_features: int, hidden_size: int, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # GRN for each feature
        self.grns = [
            GatedResidualNetwork(self.hidden_size, self.hidden_size, self.dropout_rate)
            for _ in range(self.num_features)
        ]
        
        # Softmax gate
        self.flatten = layers.Flatten()
        self.gate_dense = layers.Dense(self.num_features, activation='softmax')
        
    def call(self, x, training=None):
        # x shape: (batch, time, features)
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        # Process each feature through its GRN
        processed = []
        for i, grn in enumerate(self.grns):
            feat = x[:, :, i:i+1]  # (batch, time, 1)
            # Expand to hidden size
            feat_expanded = tf.tile(feat, [1, 1, self.hidden_size])
            processed.append(grn(feat_expanded, training=training))
        
        # Stack: (batch, time, num_features, hidden_size)
        stacked = tf.stack(processed, axis=2)
        
        # Calculate selection weights
        flat_x = self.flatten(x)  # (batch, time * features)
        weights = self.gate_dense(flat_x)  # (batch, num_features)
        weights = tf.reshape(weights, [-1, 1, self.num_features, 1])
        
        # Apply selection
        selected = tf.reduce_sum(stacked * weights, axis=2)  # (batch, time, hidden_size)
        
        return selected, weights


def build_temporal_fusion_transformer(
    input_shape: Tuple[int, int],
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_lstm_layers: int = 1,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001
) -> Model:
    """
    Build Temporal Fusion Transformer (TFT)
    Google's state-of-the-art time series model
    
    Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    """
    
    seq_len, num_features = input_shape
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # 1. Variable Selection Network
    # Select important features
    x = layers.Dense(hidden_size)(inputs)
    
    # 2. LSTM Encoder
    for i in range(num_lstm_layers):
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout_rate),
            name=f'lstm_encoder_{i}'
        )(x)
    
    # 3. Positional Encoding
    x = PositionalEncoding(seq_len, hidden_size * 2)(x)
    
    # 4. Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=hidden_size // num_attention_heads,
        dropout=dropout_rate,
        name='interpretable_attention'
    )(x, x)
    
    # Add & Norm
    x = layers.LayerNormalization()(x + attention_output)
    
    # 5. Position-wise Feed-Forward
    ff = layers.Dense(hidden_size * 4, activation='relu')(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(hidden_size * 2)(ff)
    x = layers.LayerNormalization()(x + ff)
    
    # 6. Gated Skip Connection
    gate = layers.Dense(hidden_size * 2, activation='sigmoid')(inputs[:, -1, :])
    gate = tf.expand_dims(gate, 1)
    x = x * gate
    
    # 7. Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # 8. Output layers
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='Temporal_Fusion_Transformer')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info("Built Temporal Fusion Transformer model")
    return model


def build_multi_task_model(
    input_shape: Tuple[int, int],
    hidden_size: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Model:
    """
    Build Multi-Task Learning Model
    Predicts:
    - Direction (up/down)
    - Volatility (high/low)
    - Magnitude (% change)
    """
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Shared Encoder
    x = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Bidirectional(layers.LSTM(hidden_size // 2, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Attention
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Global pooling
    shared_features = layers.GlobalAveragePooling1D()(x)
    
    # Task 1: Direction Prediction (Main task)
    direction_x = layers.Dense(64, activation='relu')(shared_features)
    direction_x = layers.Dropout(dropout_rate)(direction_x)
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(direction_x)
    
    # Task 2: Volatility Prediction
    volatility_x = layers.Dense(64, activation='relu')(shared_features)
    volatility_x = layers.Dropout(dropout_rate)(volatility_x)
    volatility_output = layers.Dense(1, activation='sigmoid', name='volatility')(volatility_x)
    
    # Task 3: Magnitude Prediction (regression)
    magnitude_x = layers.Dense(64, activation='relu')(shared_features)
    magnitude_x = layers.Dropout(dropout_rate)(magnitude_x)
    magnitude_output = layers.Dense(1, activation='linear', name='magnitude')(magnitude_x)
    
    model = Model(
        inputs=inputs,
        outputs=[direction_output, volatility_output, magnitude_output],
        name='Multi_Task_Model'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'direction': 'binary_crossentropy',
            'volatility': 'binary_crossentropy',
            'magnitude': 'mse'
        },
        loss_weights={
            'direction': 1.0,
            'volatility': 0.3,
            'magnitude': 0.2
        },
        metrics={
            'direction': ['accuracy', keras.metrics.AUC(name='auc')],
            'volatility': ['accuracy'],
            'magnitude': ['mae']
        }
    )
    
    logger.info("Built Multi-Task Learning model")
    return model


class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention Layer for modeling asset relationships
    """
    
    def __init__(self, output_dim: int, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.output_dim * self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )
        self.a = self.add_weight(
            name='attention',
            shape=(2 * self.output_dim, self.num_heads),
            initializer='glorot_uniform',
            trainable=True
        )
        
    def call(self, inputs):
        # inputs: (batch, num_nodes, features)
        # For crypto, nodes could be different assets or different timeframes
        
        h = tf.matmul(inputs, self.W)  # (batch, nodes, output_dim * heads)
        h = tf.reshape(h, [-1, tf.shape(inputs)[1], self.num_heads, self.output_dim])
        
        # Self-attention between nodes
        h_mean = tf.reduce_mean(h, axis=-1, keepdims=True)
        attention = tf.nn.softmax(h_mean, axis=1)
        
        output = tf.reduce_sum(h * attention, axis=1)  # (batch, heads, output_dim)
        output = tf.reshape(output, [-1, self.num_heads * self.output_dim])
        
        return output


def build_graph_neural_network(
    input_shape: Tuple[int, int],
    num_assets: int = 3,  # BTC, ETH, and aggregate market
    hidden_size: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Model:
    """
    Build Graph Neural Network for modeling inter-asset relationships
    """
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Treat different feature groups as "nodes" in a graph
    # Node 1: Price features
    # Node 2: Volume features  
    # Node 3: Technical indicators
    # Node 4: Market microstructure
    # Node 5: Sentiment
    
    # First, encode sequence
    x = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(inputs)
    x = layers.BatchNormalization()(x)
    
    # Split into feature groups (simulated nodes)
    num_features = input_shape[1]
    features_per_node = num_features // num_assets
    
    # Create node representations
    node_representations = []
    for i in range(num_assets):
        start_idx = i * features_per_node
        end_idx = min((i + 1) * features_per_node, num_features)
        
        node_input = inputs[:, :, start_idx:end_idx] if end_idx > start_idx else inputs
        node_x = layers.LSTM(hidden_size // num_assets, return_sequences=False)(node_input)
        node_representations.append(node_x)
    
    # Stack as graph nodes: (batch, num_nodes, hidden)
    nodes = tf.stack(node_representations, axis=1)
    
    # Graph Attention
    gat = GraphAttentionLayer(hidden_size // 2, num_heads=4)(nodes)
    
    # Combine with original sequence encoding
    seq_encoding = layers.GlobalAveragePooling1D()(x)
    combined = layers.Concatenate()([seq_encoding, gat])
    
    # Output layers
    x = layers.Dense(hidden_size, activation='relu')(combined)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_size // 2, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='Graph_Neural_Network')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info("Built Graph Neural Network model")
    return model


def build_multi_timeframe_attention_model(
    input_shape: Tuple[int, int],
    num_timeframes: int = 5,  # 5m, 15m, 1h, 4h, 1d
    hidden_size: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Model:
    """
    Build model with separate attention mechanisms for different timeframes
    Late Fusion architecture
    """
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Split features by timeframe (assuming features are organized by timeframe)
    features_per_tf = input_shape[1] // num_timeframes
    
    timeframe_encodings = []
    
    for tf_idx in range(num_timeframes):
        start_idx = tf_idx * features_per_tf
        end_idx = min((tf_idx + 1) * features_per_tf, input_shape[1])
        
        if end_idx <= start_idx:
            continue
            
        # Extract timeframe features
        tf_input = inputs[:, :, start_idx:end_idx]
        
        # Timeframe-specific encoder
        tf_x = layers.LSTM(hidden_size, return_sequences=True, 
                          name=f'lstm_tf{tf_idx}')(tf_input)
        
        # Timeframe-specific attention
        tf_attention = layers.MultiHeadAttention(
            num_heads=2, key_dim=16, name=f'attention_tf{tf_idx}'
        )(tf_x, tf_x)
        tf_x = layers.Add()([tf_x, tf_attention])
        tf_x = layers.LayerNormalization()(tf_x)
        
        # Pool to single vector
        tf_encoding = layers.GlobalAveragePooling1D()(tf_x)
        timeframe_encodings.append(tf_encoding)
    
    # Concatenate all timeframe encodings
    if len(timeframe_encodings) > 1:
        combined = layers.Concatenate()(timeframe_encodings)
    else:
        combined = timeframe_encodings[0]
    
    # Cross-timeframe attention
    # Reshape for attention: (batch, num_timeframes, hidden_size)
    num_tf = len(timeframe_encodings)
    combined_reshaped = layers.Reshape((num_tf, hidden_size))(combined)
    
    cross_attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name='cross_timeframe_attention'
    )(combined_reshaped, combined_reshaped)
    
    cross_attention = layers.Flatten()(cross_attention)
    
    # Final classification
    x = layers.Dense(hidden_size, activation='relu')(cross_attention)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_size // 2, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs, outputs, name='Multi_Timeframe_Attention')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info("Built Multi-Timeframe Attention model")
    return model


def build_advanced_model(model_type: str, input_shape: Tuple[int, int], config: Dict) -> Model:
    """Factory function to build advanced models"""
    
    builders = {
        'tft': lambda: build_temporal_fusion_transformer(
            input_shape,
            hidden_size=config.get('hidden_size', 64),
            num_attention_heads=config.get('num_heads', 4),
            num_lstm_layers=config.get('num_lstm_layers', 1),
            dropout_rate=config.get('dropout_rate', 0.1),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'multi_task': lambda: build_multi_task_model(
            input_shape,
            hidden_size=config.get('hidden_size', 128),
            dropout_rate=config.get('dropout_rate', 0.3),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'gnn': lambda: build_graph_neural_network(
            input_shape,
            num_assets=config.get('num_assets', 3),
            hidden_size=config.get('hidden_size', 64),
            dropout_rate=config.get('dropout_rate', 0.3),
            learning_rate=config.get('learning_rate', 0.001)
        ),
        'multi_tf_attention': lambda: build_multi_timeframe_attention_model(
            input_shape,
            num_timeframes=config.get('num_timeframes', 5),
            hidden_size=config.get('hidden_size', 64),
            dropout_rate=config.get('dropout_rate', 0.3),
            learning_rate=config.get('learning_rate', 0.001)
        )
    }
    
    if model_type not in builders:
        logger.warning(f"Unknown advanced model type {model_type}, defaulting to TFT")
        model_type = 'tft'
    
    return builders[model_type]()
