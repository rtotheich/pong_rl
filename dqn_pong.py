#!/usr/bin/env python

#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --time=8:00:00

# Added Callbacks, which save checkpoints every n timesteps, and recorded video every m timesteps,
# then made a reasonable guess about how many iterations could be done in 8 hours.

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.space)
    :param features_dim: (int) Number of features extracted
        This corresponds to the number of units for the last layer
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

env = make_atari_env("PongNoFrameskip-v4", seed=42)
env = VecFrameStack(env, n_stack=4)
env = VecVideoRecorder(env, "videos-v3", record_video_trigger=lambda x: x % 50000 == 0, video_length=2000)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./rich-dqn-cleanrl",
    name_prefix="rich-dqn-cleanrl",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

model = DQN("CnnPolicy",
            env,
            batch_size=32,
            buffer_size=100_000,
            learning_starts=100_000,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"runs-dqn",
            optimize_memory_usage=False,
            verbose=1,
           )
model.learn(9*1e6, callback=checkpoint_callback)
model.save("rich-dqn-cleanrl")
