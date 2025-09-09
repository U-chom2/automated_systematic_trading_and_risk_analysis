"""PPO Trading Agent implementation."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for trading environment."""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        """Initialize feature extractor.
        
        Args:
            observation_space: Environment observation space
            features_dim: Output feature dimensions
        """
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Larger network for 605 stocks
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Increased first layer
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout for better capacity
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        return self.net(observations)


class TradingCallback(BaseCallback):
    """Callback for trading agent training."""
    
    def __init__(self, verbose: int = 0):
        """Initialize callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called at each step.
        
        Returns:
            True to continue training
        """
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if len(self.locals.get('episode_rewards', [])) > 0:
            mean_reward = np.mean(self.locals['episode_rewards'])
            if self.verbose > 0:
                print(f"Mean reward: {mean_reward:.2f}")


class PPOTradingAgent:
    """PPO-based trading agent."""
    
    def __init__(
        self,
        env: gym.Env,
        num_stocks: int,
        learning_rate: float = 3e-4,
        n_steps: int = 1024,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        device: str = None,
        tensorboard_log: Optional[str] = None
    ):
        """Initialize PPO trading agent.
        
        Args:
            env: Trading environment
            num_stocks: Number of stocks to trade
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            device: Device to use
            tensorboard_log: Tensorboard log directory
        """
        self.env = env
        self.num_stocks = num_stocks
        
        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = "cuda:0"  # NVIDIA GPU
            else:
                device = "cpu"  # CPU fallback
            logger.info(f"Auto-detected device: {device}")
        
        # Convert device string to torch device
        if device == "mps":
            self.torch_device = torch.device('mps')
        elif device.startswith("cuda"):
            self.torch_device = torch.device(device)
        else:
            self.torch_device = torch.device('cpu')
        
        self.device = device
        
        # Custom policy with trading-specific features and gradient clipping
        policy_kwargs = {
            "features_extractor_class": TradingFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 512},  # Increased feature dim
            "net_arch": [dict(pi=[256, 128], vf=[256, 128])]     # Larger networks
        }
        
        # Initialize PPO model with gradient clipping
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=0.5,  # Add gradient clipping
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=tensorboard_log
        )
        
        logger.info(f"PPO agent initialized with {num_stocks} stocks on {device}")
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = True
    ):
        """Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Training callback
            progress_bar: Whether to show progress bar
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar
        )
        
        logger.info("Training completed")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted action
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating agent over {n_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode % 5 == 0:
                logger.info(f"Episode {episode}: reward={episode_reward:.2f}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        logger.info(f"Evaluation completed. Mean reward: {metrics['mean_reward']:.2f}")
        
        return metrics
    
    def save(self, path: str):
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for given observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Action probabilities
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        return probs
    
    def get_value_estimate(self, observation: np.ndarray) -> float:
        """Get value estimate for given observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Value estimate
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            value = self.model.policy.predict_values(obs_tensor).cpu().numpy()[0, 0]
        return value