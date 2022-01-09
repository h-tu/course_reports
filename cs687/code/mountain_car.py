### COMPSCI 687 Final
### Hongyu Tu

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mtc:
    discount = 1
    feature_len = 2
    action_len = 2
    inner_size = [32]
    limit = 500

    def init_d0(option = 0):
        # option 0 is train mode, 1 is test mode
        if option == 1:
            return (np.random.uniform(-0.6, -0.4), 0), False

        x, v = np.random.uniform(-1.2, 0.5), np.random.uniform(-0.07, 0.07)
        return (x, v), False

    def step(s, a):
        x, v = s
        if x == 0.5:
            return s, True, 0
        
        a = -1 if a == 0 else 1
        new_v = v + (0.001 * a - 0.0025 * np.cos(3 * x))
        new_v = min(max(-0.07, new_v), 0.07)
        new_x = x + new_v
        if new_x <= -1.2 or new_x > 0.5:
            new_v = 0
            new_x = min(max(-1.2, new_x), 0.5)
        
        reward = -1 if new_x != 0.5 else 100
        return (new_x, new_v), False, reward

    def s_tensor(s):
        x, v = s
        out = torch.tensor([x, v], dtype=torch.float, device=device)
        return out