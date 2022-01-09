### COMPSCI 687 Final
### Hongyu Tu

import torch
import numpy as np
from helper import *
# from tqdm import tqdm
from tqdm import notebook
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reinforce_w_b(env_idx, p_lr = 1e-3, v_lr = 1e-3, run = 10000, verbose = 0):
    avg_step_lst, avg_reward_lst = [], []

    # Load environment
    driver = load_env(env_idx)
    
    criteria = nn.MSELoss()
    pn = gen_net(driver.feature_len, driver.action_len, driver.inner_size)
    vn = gen_net(driver.feature_len, 1, driver.inner_size, softmax=False)

    policy_optimizer = optim.Adam(pn.parameters(), lr=p_lr)
    value_optimizer = optim.Adam(vn.parameters(), lr=v_lr)
    dis_lst = np.array([driver.discount ** j for j in range(driver.limit + 1)])
    
    for _ in notebook.tqdm(range(run)):
        pn.eval()
        vn.eval()
        count, s_lst, a_lst, r_lst = 0, [], [], []
        with torch.no_grad():
            s, done = driver.init_d0()
            while (not done) and (count < driver.limit):
                s_tmp = driver.s_tensor(s)
                # Pick action based on policy network
                action = torch.argmax(pn(s_tmp)).item()
                # Generate next step
                s, done, r = driver.step(s, action)
                # Record step info (state, action, reward)
                s_lst.append(s_tmp)
                a_lst.append(action)
                r_lst.append(r)
                count += 1
        
        if count != 0:
            # form useful lists
            tmp_dis = dis_lst[:count]
            dr_lst = np.array(r_lst) * tmp_dis
            g_lst = np.array([sum(dr_lst[idx:]) for idx in range(len(dr_lst))])
            a_lst = torch.tensor(np.array(a_lst).reshape(-1,1), dtype=torch.int64, device=device)
            s_lst = torch.stack(s_lst)

            pn.train()
            vn.train()
            
            v = vn(s_lst).view(-1)
            g = torch.tensor(g_lst, dtype=torch.float, device=device)
            v_loss = criteria(g, v)
            
            p = pn(s_lst)
            d = (g - v).detach()
            log_pi_a = torch.log(p.gather(1, a_lst)).view(-1)
            tmp_dis = torch.tensor(tmp_dis, dtype=torch.float, device=device)
            p_loss = -(tmp_dis * d * log_pi_a).sum()

            value_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            v_loss.backward()
            p_loss.backward()

            value_optimizer.step()
            policy_optimizer.step() 

        if verbose != 0 and ((_ % int(run/verbose)) == 0):
            print('[Run {}]'.format(_))
            run_policy(pn, env_idx, 20, verbose = 1)

        if _ % int(run/100) == 0:
            l, r = run_policy(pn, env_idx, 10)
            avg_step_lst.append(l)
            avg_reward_lst.append(r)

    print('[FINAL]')
    l, r = run_policy(pn, env_idx, verbose = 1)
    avg_step_lst.append(l)
    avg_reward_lst.append(r)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(avg_step_lst)) * int(run/100), avg_step_lst)
    plt.xlabel('episode')
    plt.ylabel('step length')
    plt.title('episode vs step length')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(avg_reward_lst)) * int(run/100), avg_reward_lst)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('episode vs reward')
    plt.show()
    return pn, vn


def one_ac(env_idx, p_lr = 1e-3, v_lr = 1e-3, run = 10000, verbose = 0):
    avg_step_lst, avg_reward_lst = [], []
    
    # Load environment
    driver = load_env(env_idx)
    
    criteria = nn.MSELoss()
    actor = gen_net(driver.feature_len, driver.action_len, driver.inner_size)
    critic = gen_net(driver.feature_len, 1, driver.inner_size, softmax=False)

    policy_optimizer = optim.Adam(actor.parameters(), lr=p_lr)
    value_optimizer = optim.Adam(critic.parameters(), lr=v_lr)
    dis_lst = np.array([driver.discount ** j for j in range(driver.limit + 1)])
    
    for _ in notebook.tqdm(range(run)):
        actor.eval()
        critic.eval()
        count, s_lst, a_lst, r_lst = 0, [], [], []
        with torch.no_grad():
            s, done = driver.init_d0()
            while (not done) and (count < driver.limit):
                s_tmp = driver.s_tensor(s)
                # Pick action based on policy network
                action = torch.argmax(actor(s_tmp)).item()
                # Generate next step
                s, done, r = driver.step(s, action)
                # Record step info (state, action, reward)
                s_lst.append(s_tmp)
                a_lst.append(action)
                r_lst.append(r)
                count += 1

        if count != 0:
            # form useful lists
            tmp_dis = dis_lst[:count]
            r_lst = np.array(r_lst)
            a_lst = torch.tensor(np.array(a_lst).reshape(-1,1), dtype=torch.int64, device=device)
            s_lst = torch.stack(s_lst)

            actor.train()
            critic.train()

            v = critic(s_lst).view(-1)
            v_next = torch.cat((v.view(-1)[1:], torch.zeros(1, dtype=torch.float, device=device)))
            r_tmp = torch.tensor(r_lst, dtype=torch.float, device=device)
            g = r_tmp + driver.discount * v_next
            v_loss = criteria(g, v)

            p = actor(s_lst)
            d = (g - v).detach()
            log_pi_a = torch.log(p.gather(1, a_lst)).view(-1)
            tmp_dis = torch.tensor(tmp_dis, dtype=torch.float, device=device)
            p_loss = -(tmp_dis * d * log_pi_a).sum()
            
            value_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            v_loss.backward()
            p_loss.backward()

            value_optimizer.step()
            policy_optimizer.step() 

        if verbose != 0 and ((_ % int(run/verbose)) == 0):
            print('[Run {}]'.format(_))
            run_policy(actor, env_idx, 20, verbose = 1)

        if _ % int(run/100) == 0:
            l, r = run_policy(actor, env_idx, 10)
            avg_step_lst.append(l)
            avg_reward_lst.append(r)

    print('[FINAL]')
    l, r = run_policy(actor, env_idx, verbose = 1)
    avg_step_lst.append(l)
    avg_reward_lst.append(r)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(avg_step_lst))* int(run/100), avg_step_lst)
    plt.xlabel('episode')
    plt.ylabel('step length')
    plt.title('episode vs step length')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(avg_reward_lst))* int(run/100), avg_reward_lst)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('episode vs reward')
    plt.show()
    return actor, critic 


def esg_n_sarsa(env_idx, n = 5, lr = 1e-3, epsilon = 0.1, run = 10000, verbose = 0):
    avg_step_lst, avg_reward_lst = [], []
    
    # Load environment
    driver = load_env(env_idx)
    prob_lst = (np.ones((driver.action_len, driver.action_len)) - np.eye(driver.action_len))
    prob_lst *= (epsilon/(driver.action_len - 1))
    prob_lst += np.eye(driver.action_len) * (1 - epsilon)

    criteria = nn.MSELoss()
    q_net = gen_net(driver.feature_len, driver.action_len, driver.inner_size, softmax=False)

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    dis_lst = np.array([driver.discount ** j for j in range(n + 1)])

    for _ in notebook.tqdm(range(run)):
        q_net.eval()
        count, s_lst, a_lst, r_lst = 0, [], [], []
        with torch.no_grad():
            s, done = driver.init_d0()
            while (not done) and (count < driver.limit):
                s_tmp = driver.s_tensor(s)
                
                # Pick action based on e-greedy wrt q
                q_val = q_net(s_tmp)
                max_idx = torch.argmax(q_val).item()
                action = np.random.choice(np.arange(driver.action_len), p=prob_lst[max_idx])
                
                # Generate next step
                s, done, r = driver.step(s, action)
                # Record step info (state, action, reward)
                s_lst.append(s_tmp)
                a_lst.append(action)
                r_lst.append(r)
                count += 1
            r_lst = np.array(r_lst)
            

        g_lst = []
        # form useful lists
        for i in range(count):
            end_idx = min(i + n, count)
            tmp_r_lst = list(r_lst[i:end_idx])
            if end_idx < count:
                q_est = q_net(s_lst[end_idx])[a_lst[end_idx]].item()
                tmp_r_lst.append(q_est)
            tmp_g = tmp_r_lst * dis_lst[:len(tmp_r_lst)]
            g_lst.append(np.sum(tmp_g))
        g_lst = torch.tensor(g_lst, dtype=torch.float, device=device)
        
        if count != 0:
            s_lst = torch.stack(s_lst)
            a_lst = torch.tensor(np.array(a_lst).reshape(-1,1), dtype=torch.int64, device=device)
            
            q_net.train()
            
            q_val = q_net(s_lst)
            q_val = q_val.gather(1, a_lst).view(-1)
            
            loss = criteria(g_lst, q_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose != 0 and ((_ % int(run/verbose)) == 0):
            print('[Run {}]'.format(_))
            run_policy(q_net, env_idx, 20, verbose = 1)

        if _ % int(run/100) == 0:
            l, r = run_policy(q_net, env_idx, 10)
            avg_step_lst.append(l)
            avg_reward_lst.append(r)

    print('[FINAL]')
    l, r = run_policy(q_net, env_idx, verbose = 1)
    avg_step_lst.append(l)
    avg_reward_lst.append(r)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(avg_step_lst)) * int(run/100), avg_step_lst)
    plt.xlabel('episode')
    plt.ylabel('step length')
    plt.title('episode vs step length')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(avg_reward_lst))* int(run/100), avg_reward_lst)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('episode vs reward')
    plt.show()
    return q_net