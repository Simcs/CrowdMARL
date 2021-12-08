import numpy as np
import argparse
import time
import sys
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from model import ActorCritic
from environment import make_env, make_env_pool

learning_rate = 1e-4 # learning rate
gamma = 0.99 # discount rate
gae_lambda = 0.95 # used in GAE evaluation

ppo_eps = 0.2 # clip ratio

critic_discount = 0.5 # critic loss coefficient
entropy_beta = 0.0 # 1e-3 # entropy loss coefficient

# ppo_steps = int(2048 / 4)
mini_batch_size = 64
ppo_epochs = 10

test_epochs = 10
log_epochs = 100
num_tests = 2
target_reward = -10

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

def test_env(env, model):
    state = env.reset()
    total_reward = 0
    with torch.no_grad():
        model.eval()
        for i in range(ppo_steps):
            state = torch.FloatTensor(state).to(device)
            dist, _ = model(state)
            action = dist.sample().cpu().numpy()
            next_state, reward, _ = env.step(action)
            state = next_state
            total_reward += reward
            
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=gamma, lam=gae_lambda):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

def ppo_update(frame_idx, states, actions, log_probs, returns, advantages, clip_param=ppo_eps):

    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = critic_discount * critic_loss + actor_loss - entropy_beta * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", required=True, help="Simulation time step")
    parser.add_argument("-e", "--env", default="basic", help="Environment name to use")
    parser.add_argument("-p", "--path", default="./checkpoints", help="Path to save model")
    parser.add_argument("-m", "--model", default=None, help="Model to load")
    parser.add_argument("--pool", dest="pool", action="store_true", help="Use environment pool")
    parser.add_argument("--no-pool", dest="pool", action="store_false", help="Do not use environment pool")

    parser.set_defaults(pool=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dt = float(args.dt)
    ppo_steps = int(2048 / (dt * 10))

    env = make_env(args.env, dt)
    env_pool = make_env_pool(dt)
    test = make_env(args.env, dt)
    num_inputs = env.num_observation
    num_outputs = env.num_action

    model = ActorCritic(num_inputs, num_outputs).to(device)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location=device))
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    frame_idx = 0
    train_epoch = 0
    best_reward = None

    iter = 10000
    # state = env.reset()

    start = time.time()
    for i in range(iter):
        # if args.pool and train_epoch % 10 == 0:
        #     env = random.choice(env_pool)
        #     print('current environment:', env.name)

        state = env.reset()
        model.train()

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []


        start = time.perf_counter()
        for _ in range(ppo_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            
            action = dist.sample()
            next_state, reward, done = env.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

        # print('last state:\n', state)
        print('last reward:\n', reward)
        print('step elapsed:', time.perf_counter() - start)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values
        advantages = normalize(advantages)

        print('update...')
        start = time.perf_counter()
        ppo_update(frame_idx, states, actions, log_probs, returns, advantages)
        print(f'update finished, elasped: {time.perf_counter() - start:.5f}')
        train_epoch += 1

        if train_epoch % test_epochs == 0:
            test_reward = 0
            if args.pool:
                for env_ in env_pool:
                    test_reward += np.mean([test_env(env_, model) for _ in range(num_tests)])
            else:
                test_reward = np.mean([test_env(test, model) for _ in range(num_tests)])
            print(f'Iteration {i}. avg reward: {test_reward}')
            print(f'elapsed time: {time.time() - start:.2f}')

            name = f'itr-{i},dt-{dt},reward-{test_reward:.3f}.dat'
            fname = os.path.join(args.path, name)

            if best_reward == None or best_reward < test_reward:
                best_reward = test_reward
                torch.save(model.state_dict(), fname)
            if train_epoch % log_epochs == 0:
                torch.save(model.state_dict(), fname)

