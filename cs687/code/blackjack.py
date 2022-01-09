### COMPSCI 687 Final
### Hongyu Tu

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class bjk:
    discount = 1
    action_len = 3      # 0 for wait, 1 for hit, and 2 for leave game
    feature_len = 2     # the sum of value of non-ace card, number of ace card
    inner_size = [64]    
    limit = 15

    card_range = np.arange(10) + 1
    card_prob = np.ones(10) * (4/52)
    card_prob[-1] = 16/52

    def get_val(s):
        not_a, ace = s
        val_lst = [0]
        for _ in range(ace):
            tmp = np.repeat(val_lst, 2)
            mask = np.tile([1, 10], len(val_lst))
            val_lst = tmp + mask
        val_lst = np.sort((np.unique(val_lst) + not_a))
        return val_lst[val_lst <= 21]

    def init_d0(option = 0):
        # option 0 is train mode, 1 is test mode
        if option == 1:
            return (0, 0), False
        
        s = (np.random.randint(30), np.random.randint(5))
        return s, True if len(bjk.get_val(s)) == 0 else False

    def step(s, a):
        val_lst = bjk.get_val(s)

        if len(val_lst) == 0:
            return s, True, 0

        if a == 2:
            reward = val_lst[-1]
            if reward == 21:
                reward += 5
            return s, True, reward

        reward = 0
        new_not_a, new_ace = s
        if a == 1:
            new_card = np.random.choice(bjk.card_range, p=bjk.card_prob)
            if new_card == 1:
                new_ace += 1
            else:
                new_not_a += new_card

            if len(bjk.get_val((new_not_a, new_ace))) == 0:
                reward = -5
        return (new_not_a, new_ace), False, reward

    def s_tensor(s):
        not_a, ace = s
        out = torch.tensor([not_a, ace], dtype=torch.float, device=device)
        return out