### COMPSCI 687 Final
### Hongyu Tu

### Algorithms: 0) Reinforce with Baseline
###             1) One-Step Actor-Critic
###             2) Episodic Semi-Gradient n-step SARSA
###
### Environments: 0) 687 Gridworld
###               1) Mountain Car
###               2) Blackjack
###               3) Cartpole


### Setup 
from helper import *
from algorithms import *

## Gridworld
env = 0
p0, v0 = reinforce_w_b(env)
a0, c0 = one_ac(env, run = 20000)
q0 = esg_n_sarsa(env, n = 3) 

## Mountain Car
env = 1
p1, v1 = reinforce_w_b(env, run = 100000)
a1, c1 = one_ac(env, run = 100000)
q1 = esg_n_sarsa(env, 3, run = 100000)

# Blackjack
env = 2
p2, v2 = reinforce_w_b(env)
a2, c2  = one_ac(env, run = 20000)
q2 = esg_n_sarsa(env, n = 3)

# Cartpole
env = 3
p3, v3 = reinforce_w_b(env, run = 2000)
a3, c3 = one_ac(env, run = 2500)
q3 = esg_n_sarsa(env, 3, run = 2000)

# Compare Reinforce with Baseline vs One-Step Actor-Critic
test_policy(p0, v0, 0)
test_policy(a0, c0, 0)