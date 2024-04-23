# Pong Reinforcement Learning
## Training an agent to play Pong using reinforcement learning

We train a reinforcement learning agent to play Pong using two model types: PPO and DQN.

To run, make sure to run `pip install` for the following dependencies if you do not have them installed already:

- gymnasium[atari]
- gymnasium[accept-rom-license]
- stablebaselines3
- torch

## Attribution

- PPO parameters (and video recorder idea) were inspired by <a href="https://huggingface.co/ThomasSimonini/ppo-PongNoFrameskip-v4">Thomas Simonini</a> on <a href="https://huggingface.co">Hugging Face</a>.

- DQN setup was inspired by the book <a href="https://www.oreilly.com/library/view/deep-reinforcement-learning/9781838826994/">Deep Reinforcement Learning Hands-On</a>.

- Custom policies were inspired by the <a href="https://stable-baselines3.readthedocs.io/en/master/index.html">Stablebaselines3</a> documentation on <a href="https://stable-baselines3.readthedocs.io/en/sde/guide/custom_policy.html">custom policies</a> and <a href="https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html">policy networks in general</a>.

- Neural network architecture was inspired by <a href="https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py">CleanRL</a>, who provide a high quality baseline neural network architecture for Atari games.