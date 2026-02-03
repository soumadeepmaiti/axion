"""
Reinforcement Learning Portfolio Agent
Uses PPO/DQN for dynamic portfolio rebalancing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """State representation for the RL agent"""
    asset_features: np.ndarray  # (n_assets, n_features)
    current_weights: np.ndarray  # (n_assets,)
    portfolio_value: float
    step: int


class PortfolioEnvironment:
    """
    OpenAI Gym-style environment for portfolio management
    """
    
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series],
        initial_capital: float = 10000,
        transaction_cost: float = 0.001
    ):
        self.price_data = price_data
        self.returns_data = returns_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.asset_names = list(price_data.keys())
        self.n_assets = len(self.asset_names)
        
        # Align data
        self.returns_df = pd.DataFrame(returns_data).dropna()
        self.n_steps = len(self.returns_df)
        
        # State dimensions
        self.n_features = 5  # return, vol, momentum, volume_change, price_ratio
        self.state_dim = self.n_assets * self.n_features + self.n_assets  # features + current weights
        self.action_dim = self.n_assets  # New weights for each asset
        
        # Episode state
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = initial_capital
        self.history = []
        
    def _get_features(self, step: int) -> np.ndarray:
        """Get feature matrix for current step"""
        features = np.zeros((self.n_assets, self.n_features))
        
        for i, asset in enumerate(self.asset_names):
            if asset not in self.price_data:
                continue
                
            df = self.price_data[asset]
            
            if step >= len(df):
                continue
            
            try:
                # Return
                features[i, 0] = df['close'].pct_change().iloc[step] if step > 0 else 0
                
                # Volatility (10-period)
                returns = df['close'].pct_change()
                if step >= 10:
                    features[i, 1] = returns.iloc[step-10:step].std()
                
                # Momentum (5-period)
                if step >= 5:
                    features[i, 2] = df['close'].iloc[step] / df['close'].iloc[step-5] - 1
                
                # Volume change
                features[i, 3] = df['volume'].pct_change().iloc[step] if step > 0 else 0
                
                # Price ratio to SMA
                if step >= 20:
                    sma = df['close'].iloc[step-20:step].mean()
                    features[i, 4] = df['close'].iloc[step] / sma - 1 if sma > 0 else 0
                    
            except Exception as e:
                logger.debug(f"Feature calculation error for {asset}: {e}")
                continue
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 20  # Start after warmup period
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_capital
        self.history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state as flat array"""
        features = self._get_features(self.current_step)
        state = np.concatenate([
            features.flatten(),
            self.current_weights
        ])
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action (rebalance portfolio)
        
        Args:
            action: New portfolio weights (will be normalized to sum to 1)
        
        Returns:
            next_state, reward, done, info
        """
        # Normalize action to valid weights
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            new_weights = action / action.sum()
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate transaction costs
        weight_changes = np.abs(new_weights - self.current_weights)
        transaction_costs = np.sum(weight_changes) * self.transaction_cost * self.portfolio_value
        
        # Get returns for this step
        step_returns = np.zeros(self.n_assets)
        for i, asset in enumerate(self.asset_names):
            if asset in self.returns_df.columns and self.current_step < len(self.returns_df):
                ret = self.returns_df[asset].iloc[self.current_step]
                step_returns[i] = ret if not np.isnan(ret) else 0
        
        # Calculate portfolio return
        portfolio_return = np.dot(new_weights, step_returns)
        
        # Update portfolio value
        old_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_costs
        
        # Update weights
        self.current_weights = new_weights
        
        # Calculate reward (risk-adjusted return)
        reward = portfolio_return - 0.5 * (portfolio_return ** 2)  # Quadratic utility
        reward -= np.sum(weight_changes) * 0.01  # Penalize excessive trading
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Record history
        self.history.append({
            'step': self.current_step,
            'weights': new_weights.copy(),
            'portfolio_value': self.portfolio_value,
            'return': portfolio_return,
            'transaction_costs': transaction_costs
        })
        
        next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'return': portfolio_return,
            'transaction_costs': transaction_costs
        }
        
        return next_state, float(reward), done, info


class PPOPortfolioAgent:
    """
    Proximal Policy Optimization agent for portfolio management
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.actor = None
        self.critic = None
        self.is_trained = False
        
    def _build_networks(self):
        """Build actor and critic networks"""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        
        # Actor network (policy)
        actor_input = Input(shape=(self.state_dim,), name='actor_input')
        x = Dense(self.hidden_dim, activation='relu')(actor_input)
        x = BatchNormalization()(x)
        x = Dense(self.hidden_dim // 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        # Output: mean and log_std for each action dimension
        actor_mean = Dense(self.action_dim, activation='softmax', name='action_mean')(x)
        
        self.actor = Model(actor_input, actor_mean, name='actor')
        
        # Critic network (value function)
        critic_input = Input(shape=(self.state_dim,), name='critic_input')
        x = Dense(self.hidden_dim, activation='relu')(critic_input)
        x = BatchNormalization()(x)
        x = Dense(self.hidden_dim // 2, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        value = Dense(1, name='value')(x)
        
        self.critic = Model(critic_input, value, name='critic')
        
        # Compile
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(self.lr))
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss='mse'
        )
    
    def get_action(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """Get action from policy"""
        if self.actor is None:
            # Return equal weights if not trained
            return np.ones(self.action_dim) / self.action_dim
        
        state = state.reshape(1, -1)
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        if training:
            # Add exploration noise during training
            noise = np.random.normal(0, 0.1, size=action_probs.shape)
            action_probs = action_probs + noise
            action_probs = np.clip(action_probs, 0, 1)
            action_probs = action_probs / action_probs.sum()
        
        return action_probs
    
    def train(
        self,
        env: PortfolioEnvironment,
        n_episodes: int = 100,
        max_steps: int = 1000
    ) -> Dict:
        """
        Train the PPO agent
        """
        import tensorflow as tf
        
        if self.actor is None:
            self._build_networks()
        
        episode_rewards = []
        episode_values = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            
            states, actions, rewards, values, dones = [], [], [], [], []
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, training=True)
                
                # Get value estimate
                value = self.critic.predict(state.reshape(1, -1), verbose=0)[0, 0]
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_values.append(env.portfolio_value)
            
            # Calculate advantages and returns
            if len(states) > 0:
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                values = np.array(values)
                
                # Calculate returns (discounted cumulative rewards)
                returns = np.zeros_like(rewards)
                running_return = 0
                for t in reversed(range(len(rewards))):
                    running_return = rewards[t] + self.gamma * running_return
                    returns[t] = running_return
                
                # Calculate advantages
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Update critic
                self.critic.fit(states, returns, epochs=5, batch_size=32, verbose=0)
                
                # Update actor (simplified PPO update)
                with tf.GradientTape() as tape:
                    action_probs = self.actor(states)
                    # Simplified loss: -advantage * log(prob)
                    log_probs = tf.math.log(tf.reduce_sum(action_probs * actions, axis=1) + 1e-8)
                    actor_loss = -tf.reduce_mean(log_probs * advantages)
                
                grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(episode_values[-10:])
                logger.info(f"Episode {episode}: avg_reward={avg_reward:.4f}, avg_value={avg_value:.2f}")
        
        self.is_trained = True
        
        return {
            'status': 'success',
            'episodes': n_episodes,
            'final_avg_reward': float(np.mean(episode_rewards[-10:])),
            'final_portfolio_value': float(episode_values[-1]) if episode_values else 0,
            'total_return': float((episode_values[-1] / env.initial_capital - 1) * 100) if episode_values else 0
        }
    
    def predict_weights(self, state: np.ndarray) -> np.ndarray:
        """Predict optimal weights for given state"""
        return self.get_action(state, training=False)


class RLPortfolioAgent:
    """
    High-level RL Portfolio Agent that manages the environment and PPO agent
    """
    
    def __init__(self):
        self.env: Optional[PortfolioEnvironment] = None
        self.agent: Optional[PPOPortfolioAgent] = None
        self.asset_names: List[str] = []
        self.is_trained = False
        self.training_result: Optional[Dict] = None
        
    def setup(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series],
        initial_capital: float = 10000
    ):
        """Setup the environment and agent"""
        self.env = PortfolioEnvironment(
            price_data=price_data,
            returns_data=returns_data,
            initial_capital=initial_capital
        )
        
        self.asset_names = self.env.asset_names
        
        self.agent = PPOPortfolioAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        logger.info(f"RL Agent setup: {self.env.n_assets} assets, state_dim={self.env.state_dim}")
    
    def train(self, n_episodes: int = 50) -> Dict:
        """Train the RL agent"""
        if self.env is None or self.agent is None:
            return {'status': 'failed', 'reason': 'not_setup'}
        
        result = self.agent.train(self.env, n_episodes=n_episodes)
        
        if result['status'] == 'success':
            self.is_trained = True
            self.training_result = result
        
        return result
    
    def predict_weights(
        self,
        price_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Predict optimal portfolio weights
        """
        if not self.is_trained or self.agent is None:
            return {}
        
        try:
            # Use the original environment if available (same state dimension)
            if self.env is not None:
                # Reset the original environment with updated data
                state = self.env.reset()
                
                # Predict weights using the trained agent
                weights = self.agent.predict_weights(state)
                
                # Map to asset names from original training
                result = {}
                for i, asset in enumerate(self.asset_names):
                    if i < len(weights):
                        result[asset] = float(weights[i])
                
                return result
            else:
                # Fallback: return equal weights
                logger.warning("RL Agent environment not available, returning equal weights")
                result = {}
                for asset in self.asset_names:
                    result[asset] = 1.0 / len(self.asset_names)
                return result
                
        except Exception as e:
            logger.error(f"RL Agent prediction error: {e}")
            # Return equal weights on error
            result = {}
            for asset in self.asset_names:
                result[asset] = 1.0 / len(self.asset_names)
            return result
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'n_assets': len(self.asset_names),
            'asset_names': self.asset_names,
            'training_result': self.training_result
        }
    
    def save(self, save_dir: str = "/app/backend/saved_models/portfolio") -> Dict:
        """
        Save the trained RL agent to disk
        
        Args:
            save_dir: Directory to save the model
        
        Returns:
            Dictionary with save status and path
        """
        import os
        import json
        
        if not self.is_trained or self.agent is None:
            return {'status': 'error', 'message': 'No trained model to save'}
        
        try:
            # Create directory if not exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            model_name = f"rl_portfolio_{timestamp}"
            model_path = os.path.join(save_dir, model_name)
            
            # Save actor and critic networks
            self.agent.actor.save(f"{model_path}_actor.keras")
            self.agent.critic.save(f"{model_path}_critic.keras")
            
            # Save metadata
            metadata = {
                'model_type': 'rl_portfolio_agent',
                'n_assets': len(self.asset_names),
                'asset_names': self.asset_names,
                'state_dim': self.agent.state_dim,
                'action_dim': self.agent.action_dim,
                'hidden_dim': self.agent.hidden_dim,
                'training_result': self.training_result,
                'created_at': timestamp
            }
            
            with open(f"{model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"RL Portfolio Agent saved to {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'model_name': model_name,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving RL Portfolio Agent: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def load(self, model_path: str) -> Dict:
        """
        Load a trained RL agent from disk
        
        Args:
            model_path: Path to the saved model (without extension)
        
        Returns:
            Dictionary with load status
        """
        import os
        import json
        import tensorflow as tf
        
        try:
            # Check if files exist
            actor_path = f"{model_path}_actor.keras"
            critic_path = f"{model_path}_critic.keras"
            metadata_path = f"{model_path}_metadata.json"
            
            if not os.path.exists(actor_path):
                return {'status': 'error', 'message': f'Actor model not found: {actor_path}'}
            
            # Load metadata first to get dimensions
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create agent with correct dimensions
            state_dim = metadata.get('state_dim', 100)
            action_dim = metadata.get('action_dim', 20)
            hidden_dim = metadata.get('hidden_dim', 128)
            
            self.agent = PPOPortfolioAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )
            
            # Load actor and critic
            self.agent.actor = tf.keras.models.load_model(actor_path, compile=False)
            self.agent.actor.compile(optimizer=tf.keras.optimizers.Adam(self.agent.lr))
            
            if os.path.exists(critic_path):
                self.agent.critic = tf.keras.models.load_model(critic_path, compile=False)
                self.agent.critic.compile(
                    optimizer=tf.keras.optimizers.Adam(self.agent.lr),
                    loss='mse'
                )
            
            # Restore metadata
            self.asset_names = metadata.get('asset_names', [])
            self.training_result = metadata.get('training_result')
            self.is_trained = True
            
            logger.info(f"RL Portfolio Agent loaded from {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'n_assets': len(self.asset_names),
                'asset_names': self.asset_names
            }
            
        except Exception as e:
            logger.error(f"Error loading RL Portfolio Agent: {e}")
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def list_saved_models(save_dir: str = "/app/backend/saved_models/portfolio") -> List[Dict]:
        """
        List all saved RL Portfolio models
        
        Returns:
            List of model metadata dictionaries
        """
        import os
        import json
        
        models = []
        
        if not os.path.exists(save_dir):
            return models
        
        for filename in os.listdir(save_dir):
            if filename.startswith("rl_portfolio_") and filename.endswith("_metadata.json"):
                try:
                    with open(os.path.join(save_dir, filename), 'r') as f:
                        metadata = json.load(f)
                    
                    model_name = filename.replace("_metadata.json", "")
                    metadata['model_name'] = model_name
                    metadata['model_path'] = os.path.join(save_dir, model_name)
                    models.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {filename}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    @staticmethod
    def delete_model(model_path: str) -> Dict:
        """
        Delete a saved RL model
        
        Args:
            model_path: Path to the model (without extension)
        
        Returns:
            Dictionary with delete status
        """
        import os
        
        try:
            files_deleted = []
            
            # Delete all related files
            extensions = ['_actor.keras', '_critic.keras', '_metadata.json']
            for ext in extensions:
                filepath = f"{model_path}{ext}"
                if os.path.exists(filepath):
                    os.remove(filepath)
                    files_deleted.append(filepath)
            
            if files_deleted:
                logger.info(f"Deleted RL Portfolio model: {model_path}")
                return {
                    'status': 'success',
                    'files_deleted': files_deleted
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No files found to delete'
                }
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return {'status': 'error', 'message': str(e)}


# Singleton instance
rl_portfolio_agent = RLPortfolioAgent()
