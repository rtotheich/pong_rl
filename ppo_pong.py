#!/usr/bin/env python

#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --time=8:00:00

# Added Callbacks, which save checkpoints every n timesteps, and recorded video every m timesteps,
# then made a reasonable guess about how many iterations could be done in 8 hours.

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
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
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

env = make_atari_env("PongNoFrameskip-v4", seed=42)
env = VecFrameStack(env, n_stack=4)
env = VecVideoRecorder(env, "videos-ppo", record_video_trigger=lambda x: x % 50000 == 0, video_length=2000)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./rich-cleanrl-ppo",
    name_prefix="rich-cleanrl-ppo",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

model = PPO("CnnPolicy",
            env,
            batch_size=256,
            clip_range=0.1,
            ent_coef=0.01,
            gae_lambda=0.9,
            gamma=0.99,
            learning_rate=2.5e-4,
            max_grad_norm=0.5,
            n_epochs=4,
            n_steps=128,
            vf_coef=0.5,
            tensorboard_log=f"runs-ppo",
            verbose=1
           )
model.learn(9*1e6, callback=checkpoint_callback)
model.save("rich-cleanrl-ppo")
