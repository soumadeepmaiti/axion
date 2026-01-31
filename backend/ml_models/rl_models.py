"""
Reinforcement Learning Models for Crypto Trading

Implements:
- DQN (Deep Q-Network) - For discrete action space (BUY/HOLD/SELL)
- PPO (Proximal Policy Optimization) - State-of-the-art policy gradient
- A2C (Advantage Actor-Critic) - Actor-Critic method

The RL agent learns to maximize cumulative trading rewards through
trial and error, discovering optimal trading strategies.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
from collections import deque
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Actions
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
NUM_ACTIONS = 3


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class TradingEnvironment:
    """
    Trading environment for RL training
    
    State: Window of market features (OHLCV + indicators)
    Actions: HOLD (0), BUY (1), SELL (2)
    Reward: PnL from action + risk-adjusted bonus
    """
    
    def __init__(self, data: np.ndarray, prices: np.ndarray, 
                 window_size: int = 50, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001):
        self.data = data
        self.prices = prices
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state (market window + position info)"""
        market_state = self.data[self.current_step - self.window_size:self.current_step]
        
        # Add position information to state
        position_info = np.array([[self.position, self.entry_price / (self.prices[self.current_step] + 1e-8)]])
        position_broadcast = np.repeat(position_info, self.window_size, axis=0)
        
        return np.concatenate([market_state, position_broadcast], axis=1)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info)
        """
        current_price = self.prices[self.current_step]
        reward = 0.0
        info = {}
        
        # Execute action
        if action == ACTION_BUY and self.position <= 0:
            # Open long or close short
            if self.position < 0:  # Close short
                pnl = (self.entry_price - current_price) * abs(self.position) * self.balance
                pnl -= self.balance * self.transaction_cost  # Transaction cost
                reward = pnl / self.initial_balance
                self.total_pnl += pnl
                self.trades.append({'type': 'close_short', 'price': current_price, 'pnl': pnl})
            
            # Open long
            self.position = 1.0
            self.entry_price = current_price
            self.balance -= self.balance * self.transaction_cost
            info['action_taken'] = 'BUY'
            
        elif action == ACTION_SELL and self.position >= 0:
            # Open short or close long
            if self.position > 0:  # Close long
                pnl = (current_price - self.entry_price) * self.position * self.balance
                pnl -= self.balance * self.transaction_cost
                reward = pnl / self.initial_balance
                self.total_pnl += pnl
                self.trades.append({'type': 'close_long', 'price': current_price, 'pnl': pnl})
            
            # Open short
            self.position = -1.0
            self.entry_price = current_price
            self.balance -= self.balance * self.transaction_cost
            info['action_taken'] = 'SELL'
            
        else:
            # HOLD - small penalty to encourage action
            reward = -0.0001
            info['action_taken'] = 'HOLD'
            
            # Unrealized PnL for holding position
            if self.position > 0:
                unrealized = (current_price - self.entry_price) / self.entry_price
                reward += unrealized * 0.1  # Small reward for good positions
            elif self.position < 0:
                unrealized = (self.entry_price - current_price) / self.entry_price
                reward += unrealized * 0.1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Final reward adjustment
        if done and self.position != 0:
            # Force close any open position
            final_price = self.prices[self.current_step]
            if self.position > 0:
                pnl = (final_price - self.entry_price) * self.position * self.balance
            else:
                pnl = (self.entry_price - final_price) * abs(self.position) * self.balance
            reward += pnl / self.initial_balance
            self.total_pnl += pnl
        
        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        
        info['total_pnl'] = self.total_pnl
        info['balance'] = self.balance + self.total_pnl
        info['num_trades'] = len(self.trades)
        
        return next_state, reward, done, info


def build_dqn_network(state_shape: Tuple[int, int], num_actions: int = 3,
                      hidden_units: List[int] = [256, 128, 64]) -> Model:
    """
    Build Deep Q-Network for trading
    
    Architecture: LSTM encoder + Dense Q-value head
    """
    inputs = layers.Input(shape=state_shape, name='state_input')
    
    # LSTM encoder for temporal patterns
    x = layers.LSTM(hidden_units[0], return_sequences=True, name='lstm_1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(hidden_units[1], return_sequences=False, name='lstm_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Dense layers for Q-value estimation
    for i, units in enumerate(hidden_units[2:], start=1):
        x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = layers.Dropout(0.2)(x)
    
    # Dueling DQN architecture
    # Value stream
    value = layers.Dense(64, activation='relu', name='value_hidden')(x)
    value = layers.Dense(1, name='value')(value)
    
    # Advantage stream
    advantage = layers.Dense(64, activation='relu', name='advantage_hidden')(x)
    advantage = layers.Dense(num_actions, name='advantage')(advantage)
    
    # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    q_values = layers.Lambda(
        lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
        name='q_values'
    )([value, advantage])
    
    model = Model(inputs, q_values, name='DQN_Trading')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber'
    )
    
    return model


def build_actor_critic_network(state_shape: Tuple[int, int], num_actions: int = 3,
                               hidden_units: List[int] = [256, 128]) -> Tuple[Model, Model]:
    """
    Build Actor-Critic networks for PPO/A2C
    
    Actor: Policy network π(a|s)
    Critic: Value network V(s)
    """
    # Shared encoder
    state_input = layers.Input(shape=state_shape, name='state_input')
    
    x = layers.LSTM(hidden_units[0], return_sequences=True, name='shared_lstm_1')(state_input)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(hidden_units[1], return_sequences=False, name='shared_lstm_2')(x)
    x = layers.BatchNormalization()(x)
    
    # Actor (policy) network
    actor_x = layers.Dense(128, activation='relu', name='actor_dense_1')(x)
    actor_x = layers.Dense(64, activation='relu', name='actor_dense_2')(actor_x)
    action_probs = layers.Dense(num_actions, activation='softmax', name='action_probs')(actor_x)
    
    actor = Model(state_input, action_probs, name='Actor_Policy')
    actor.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='categorical_crossentropy')
    
    # Critic (value) network
    critic_x = layers.Dense(128, activation='relu', name='critic_dense_1')(x)
    critic_x = layers.Dense(64, activation='relu', name='critic_dense_2')(critic_x)
    value = layers.Dense(1, name='state_value')(critic_x)
    
    critic = Model(state_input, value, name='Critic_Value')
    critic.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    return actor, critic


class DQNAgent:
    """
    Deep Q-Network Agent for Trading
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Dueling architecture
    """
    
    def __init__(self, state_shape: Tuple[int, int], num_actions: int = 3,
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update_freq: int = 100):
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = build_dqn_network(state_shape, num_actions)
        self.target_network = build_dqn_network(state_shape, num_actions)
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.training_step = 0
        self.is_trained = False
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        state = np.expand_dims(state, 0)
        q_values = self.q_network.predict(state, verbose=0)[0]
        return int(np.argmax(q_values))
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(Experience(state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Compute target Q-values (Double DQN)
        next_q_values = self.q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values, axis=1)
        
        target_q_values = self.target_network.predict(next_states, verbose=0)
        target_q = rewards + self.gamma * target_q_values[np.arange(len(batch)), next_actions] * (1 - dones)
        
        # Update Q-network
        current_q = self.q_network.predict(states, verbose=0)
        current_q[np.arange(len(batch)), actions] = target_q
        
        loss = self.q_network.train_on_batch(states, current_q)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return float(loss)
    
    def save(self, path: str):
        """Save agent weights"""
        self.q_network.save(f"{path}/q_network.keras")
        self.target_network.save(f"{path}/target_network.keras")
        logger.info(f"DQN agent saved to {path}")
    
    def load(self, path: str):
        """Load agent weights"""
        self.q_network = keras.models.load_model(f"{path}/q_network.keras")
        self.target_network = keras.models.load_model(f"{path}/target_network.keras")
        self.is_trained = True
        logger.info(f"DQN agent loaded from {path}")


class PPOAgent:
    """
    Proximal Policy Optimization Agent for Trading
    
    Features:
    - Clipped surrogate objective
    - Advantage estimation (GAE)
    - Entropy bonus for exploration
    """
    
    def __init__(self, state_shape: Tuple[int, int], num_actions: int = 3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, lr_actor: float = 0.0003,
                 lr_critic: float = 0.001):
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Networks
        self.actor, self.critic = build_actor_critic_network(state_shape, num_actions)
        self.actor.optimizer = keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic.optimizer = keras.optimizers.Adam(learning_rate=lr_critic)
        
        self.is_trained = False
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action from policy"""
        state = np.expand_dims(state, 0)
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        if training:
            action = np.random.choice(self.num_actions, p=action_probs)
        else:
            action = int(np.argmax(action_probs))
        
        return action, action_probs[action]
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns"""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, 
                  old_probs: np.ndarray, advantages: np.ndarray,
                  returns: np.ndarray, epochs: int = 4) -> Dict:
        """PPO training step with clipped objective"""
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(epochs):
            # Actor update
            with tf.GradientTape() as tape:
                action_probs = self.actor(states, training=True)
                action_indices = tf.stack([tf.range(len(actions)), actions], axis=1)
                new_probs = tf.gather_nd(action_probs, action_indices)
                
                ratio = new_probs / (old_probs + 1e-8)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                surrogate1 = ratio * advantages
                surrogate2 = clipped_ratio * advantages
                
                # Entropy bonus
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
                
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                actor_loss -= self.entropy_coef * tf.reduce_mean(entropy)
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            actor_losses.append(float(actor_loss))
            
            # Critic update
            with tf.GradientTape() as tape:
                values = self.critic(states, training=True)
                critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))
            
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            critic_losses.append(float(critic_loss))
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }
    
    def save(self, path: str):
        """Save agent weights"""
        self.actor.save(f"{path}/actor.keras")
        self.critic.save(f"{path}/critic.keras")
        logger.info(f"PPO agent saved to {path}")
    
    def load(self, path: str):
        """Load agent weights"""
        self.actor = keras.models.load_model(f"{path}/actor.keras")
        self.critic = keras.models.load_model(f"{path}/critic.keras")
        self.is_trained = True
        logger.info(f"PPO agent loaded from {path}")


class RLTrainer:
    """
    Reinforcement Learning Trainer for trading models
    
    Supports both DQN and PPO training
    """
    
    def __init__(self, algorithm: str = 'dqn', config: Dict = None):
        self.algorithm = algorithm
        self.config = config or {}
        self.agent = None
        self.env = None
        self.is_trained = False
        self.training_history = []
    
    def setup(self, data: np.ndarray, prices: np.ndarray, window_size: int = 50):
        """Setup environment and agent"""
        self.env = TradingEnvironment(data, prices, window_size)
        
        state_shape = (window_size, data.shape[1] + 2)  # +2 for position info
        
        if self.algorithm == 'dqn':
            self.agent = DQNAgent(
                state_shape,
                gamma=self.config.get('gamma', 0.99),
                epsilon=self.config.get('epsilon', 1.0),
                epsilon_decay=self.config.get('epsilon_decay', 0.995),
                batch_size=self.config.get('batch_size', 64)
            )
        elif self.algorithm == 'ppo':
            self.agent = PPOAgent(
                state_shape,
                gamma=self.config.get('gamma', 0.99),
                clip_ratio=self.config.get('clip_ratio', 0.2),
                entropy_coef=self.config.get('entropy_coef', 0.01)
            )
        else:
            raise ValueError(f"Unknown RL algorithm: {self.algorithm}")
        
        logger.info(f"RL Trainer initialized with {self.algorithm.upper()} algorithm")
    
    def train(self, num_episodes: int = 100, 
              status_callback=None) -> Dict:
        """Train the RL agent"""
        
        if self.agent is None or self.env is None:
            raise ValueError("Trainer not setup. Call setup() first.")
        
        episode_rewards = []
        episode_pnls = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            episode_data = {'states': [], 'actions': [], 'rewards': [], 
                           'probs': [], 'values': [], 'dones': []}
            
            while True:
                if self.algorithm == 'dqn':
                    action = self.agent.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    
                    self.agent.store_experience(state, action, reward, next_state, done)
                    loss = self.agent.train_step()
                    
                elif self.algorithm == 'ppo':
                    action, prob = self.agent.select_action(state)
                    value = float(self.agent.critic.predict(np.expand_dims(state, 0), verbose=0)[0])
                    
                    next_state, reward, done, info = self.env.step(action)
                    
                    episode_data['states'].append(state)
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['probs'].append(prob)
                    episode_data['values'].append(value)
                    episode_data['dones'].append(done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # PPO update at end of episode
            if self.algorithm == 'ppo' and len(episode_data['states']) > 0:
                states = np.array(episode_data['states'])
                actions = np.array(episode_data['actions'], dtype=np.int32)
                old_probs = np.array(episode_data['probs'])
                
                advantages, returns = self.agent.compute_advantages(
                    episode_data['rewards'],
                    episode_data['values'],
                    episode_data['dones']
                )
                
                self.agent.train_step(states, actions, old_probs, advantages, returns)
            
            episode_rewards.append(total_reward)
            episode_pnls.append(info['total_pnl'])
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_pnl = np.mean(episode_pnls[-10:])
                logger.info(f"Episode {episode + 1}/{num_episodes} - "
                           f"Avg Reward: {avg_reward:.4f}, Avg PnL: ${avg_pnl:.2f}")
                
                if status_callback:
                    status_callback({
                        'episode': episode + 1,
                        'total_episodes': num_episodes,
                        'avg_reward': avg_reward,
                        'avg_pnl': avg_pnl,
                        'epsilon': getattr(self.agent, 'epsilon', None)
                    })
            
            self.training_history.append({
                'episode': episode + 1,
                'reward': total_reward,
                'pnl': info['total_pnl'],
                'num_trades': info['num_trades']
            })
        
        self.is_trained = True
        
        return {
            'final_avg_reward': np.mean(episode_rewards[-10:]),
            'final_avg_pnl': np.mean(episode_pnls[-10:]),
            'best_pnl': max(episode_pnls),
            'total_episodes': num_episodes,
            'algorithm': self.algorithm
        }
    
    def predict(self, state: np.ndarray) -> Dict:
        """Get trading action from trained agent"""
        if not self.is_trained:
            return {'action': ACTION_HOLD, 'action_name': 'HOLD', 'confidence': 0.5}
        
        if self.algorithm == 'dqn':
            action = self.agent.select_action(state, training=False)
            q_values = self.agent.q_network.predict(np.expand_dims(state, 0), verbose=0)[0]
            confidence = float(np.max(q_values) - np.min(q_values)) / (abs(np.max(q_values)) + 1e-8)
        else:
            action, prob = self.agent.select_action(state, training=False)
            confidence = float(prob)
        
        action_names = {ACTION_HOLD: 'HOLD', ACTION_BUY: 'BUY', ACTION_SELL: 'SELL'}
        
        return {
            'action': action,
            'action_name': action_names[action],
            'confidence': min(confidence, 1.0),
            'direction': 1 if action == ACTION_BUY else (0 if action == ACTION_SELL else 0.5)
        }
    
    def save(self, path: str):
        """Save trained agent"""
        if self.agent:
            self.agent.save(path)
    
    def load(self, path: str):
        """Load trained agent"""
        if self.agent:
            self.agent.load(path)
            self.is_trained = True


# Factory function for building RL models
def build_rl_model(algorithm: str, state_shape: Tuple[int, int], config: Dict = None):
    """Build RL model based on algorithm type"""
    config = config or {}
    
    if algorithm == 'dqn':
        return DQNAgent(
            state_shape,
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 1.0),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            batch_size=config.get('batch_size', 64)
        )
    elif algorithm == 'ppo':
        return PPOAgent(
            state_shape,
            gamma=config.get('gamma', 0.99),
            clip_ratio=config.get('clip_ratio', 0.2)
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {algorithm}")
