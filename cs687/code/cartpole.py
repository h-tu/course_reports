### COMPSCI 687 Final
### Hongyu Tu

import gym
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')

class cpl:
    discount = 1
    feature_len = 4
    action_len = 2
    inner_size = [24,48]
    limit = 200

    def init_d0(option=0):
        return env.reset(), False

    def step(_, a):
       observation, reward, done, _ = env.step(a)
       if done == True:
           reward = -50
       return observation, done, reward

    def s_tensor(s):
        out = torch.tensor(s, dtype=torch.float, device=device)
        return out