import torch
from helper import *

model = torch.load('cartpole_500.pt')
model.eval()

test_policy(model, None, 3)