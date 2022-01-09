### COMPSCI 687 Final
### Hongyu Tu

import gym
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arrow = ['↑', '↓', '←', '→']
lib_lst = [('gridworld', 'gdw'), \
           ('mountain_car', 'mtc'), \
           ('blackjack', 'bjk'), \
		   ('cartpole', 'cpl')]

optimal_policy = np.array([[3, 3, 3, 1, 1],
                           [3, 3, 3, 1, 1],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 1, 1],
                           [0, 0, 3, 3, 0]])

optimal_value = np.array([[4.01868872, 4.55478377, 5.1575445 , 5.83363576, 6.45528789],
                          [4.37160489, 5.03235842, 5.80129548, 6.64726542, 7.39070891],
                          [3.86716833, 4.38996463, 0.        , 7.57690464, 8.46366148],
                          [3.41825098, 3.83189717, 0.        , 8.5738302 , 9.69459232],
                          [2.99769474, 2.93093024, 6.07330058, 9.69459232, 0.        ]])

def load_env(env_idx):
    return getattr(__import__(lib_lst[env_idx][0]), lib_lst[env_idx][1])

def print_mat(mat):
    if 'float' in str(mat.dtype):
        name = 'Value Function'
        str_lst = np.array(['{:.4f}'.format(i) for i in mat.reshape(-1)]).reshape(5,5)
    else:
        name = 'Policy'
        str_lst = np.array([arrow[i] for i in mat.reshape(-1)]).reshape(5,5)
        str_lst[2:4,2], str_lst[4,4] = ' ', 'G'
    print(name)
    for row in str_lst:
        print('   '.join(row))
    print()


def gen_net(input_len, output_len, middle = [128], softmax = True):
    layers = [nn.Linear(input_len, middle[0]), nn.ReLU()]
    for i in range(len(middle) - 1):
        layers.append(nn.Linear(middle[i], middle[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(middle[-1], output_len))
    if softmax:
        layers.append(nn.Softmax(dim=-1))

    model = nn.Sequential(*layers).to(device)
    return model


def test_policy(p_net, v_net, env):
    p_net.eval()
    if env != 3:
        v_net.eval()  

    driver = load_env(env)

    if env == 0:
        v_mat = np.zeros((5,5))
        p_mat = np.zeros((5,5), dtype = 'int')

        for i in range(5):
            for j in range(5):
                v_mat[i,j] = v_net(driver.s_tensor((i,j))).item()
                p_mat[i,j] = int(torch.argmax(p_net(driver.s_tensor((i,j)))).item())

        print_mat(p_mat)
        print_mat(v_mat)
    elif env == 1:
        x = np.linspace(-1.2, 0.5, 1000)
        b = torch.stack((torch.linspace(-1.2, 0.5, 1000, device = device), \
                         torch.zeros(1000, device = device)))
        b = torch.transpose(b, 0, 1).view(-1, 2)
        y = v_net(b).cpu().detach().numpy()
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('v(x,0)')
        plt.title('x vs v(x,0)')
        plt.show()
    elif env == 3:
        tmp_env = gym.make('CartPole-v1')
        for i_episode in range(10):
            observation = tmp_env.reset()
            t, done = 0, False
            while not done:
                tmp_env.render()
                # print(observation)
                action = torch.argmax(p_net(driver.s_tensor(observation))).item()
                observation, reward, done, info = tmp_env.step(action)
                t += 1
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break

def run_policy(p_net, env, run = 100, verbose = 0):
    driver = load_env(env)

    step_lst = []
    return_lst = []
    p_net.eval()

    for _ in range(run):
        count, j = 0, 0
        with torch.no_grad():
            s, done = driver.init_d0(1)
            while (not done) and (count < driver.limit):
                a = torch.argmax(p_net(driver.s_tensor(s))).item()
                s, done, r = driver.step(s, a)
                j += driver.discount ** count * r
                count += 1
        step_lst.append(count)
        return_lst.append(j)

    avg_len, avg_return = np.mean(step_lst), np.mean(return_lst)
    if verbose == 1:
        print('The average episode length is {:.2f}\nThe average return is {:.4f}\n'.format(avg_len, avg_return))
    return avg_len, avg_return