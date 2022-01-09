### COMPSCI 687 Final
### Hongyu Tu

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class gdw:
    discount = 0.9
    feature_len = 10 # 5 from one-hot encoded r and 5 from one-hot encoded c
    action_len = 4
    inner_size = [128]
    limit = 50

    def init_d0(option = 0):
        # option 0 is train mode, 1 is test mode
        if option == 1:
            return (0, 0), False
        
        s = np.random.randint(1,24)
        s = s + 2 if s > 16 else s
        s = s + 1 if s > 12 and s < 17 else s
        r = int((s - 1) / 5)
        c = (s - 1) % 5 
        return (r,c), False

    def step(s, a):
        if s == (4,4):
            return (4,4), True, 0

        r,c = s
        action = (a, np.random.choice([0, 1, 2, 3], p=[0.8, 0.05, 0.05, 0.1]))
        
        new_r, new_c = r, c
        if (action == (0,0) or action == (2,2) or action == (3,1)) and r > 0:      # Up
            new_r = r - 1
        elif (action == (0,1) or action == (1,2) or action == (2,0)) and c > 0:    # Left
            new_c = c - 1
        elif (action == (0,2) or action == (1,1) or action == (3,0)) and c < 4:    # Right
            new_c = c + 1
        elif (action == (1,0) or action == (2,1) or action == (3,2)) and r < 4:    # Down
            new_r = r + 1
        if (new_r, new_c) == (2,2) or (new_r, new_c) == (3,2):
            new_r, new_c = r, c
        
        reward = 0
        if (new_r, new_c) == (4,2):                   # reward for water
            reward = -10           
        elif (new_r, new_c) == (4,4):                 # reward for first arrival at terminal
            reward = 10
        return (new_r, new_c), False, reward

    def s_tensor(s):
        tmp = np.concatenate((np.eye(5)[s[0]], np.eye(5)[s[1]]))
        out = torch.tensor(tmp, dtype=torch.float, device=device)
        return out