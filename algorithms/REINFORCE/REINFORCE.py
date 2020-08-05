# Paper: https://arxiv.org/pdf/1604.06778.pdf
# Got the basic idea from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
# Solving Cartpole-v1 with 472 episodes
import gym
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
import numpy as np

from algorithms.REINFORCE.config import arg_parser

EPS = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self, obs_shape=0, act_shape=0):
        super(Policy, self).__init__()
        assert obs_shape > 0 and act_shape > 0

        self.layer_1 = nn.Linear(obs_shape, 128)
        self.drop_out_1 = nn.Dropout(p=0.6)
        self.layer_2 = nn.Linear(128, act_shape)

        self.rewards = []
        self.log_prob = []

    def forward(self, obs):
        x = self.layer_1(obs)
        x = self.drop_out_1(x)
        x = F.relu(x)
        act_prob = self.layer_2(x)
        act_prob = F.softmax(act_prob, dim=1)
        return act_prob


def select_action(obs, policy):
    # 1. change numpy,array to tensor, 2. ensure it is float, 3. unsqueeze, since nn.Module can only deal with
    # mini-batch, wich means even we only have one observation, we still need to unsqueeze it. change from CHW -> BCHW
    obs = torch.from_numpy(obs).float().unsqueeze(dim=0)
    act_prob = policy(obs)

    # https://pytorch.org/docs/stable/distributions.html
    # Categorical has sample() and log_prob()
    m = Categorical(act_prob)
    action = m.sample()
    policy.log_prob.append(m.log_prob(action))

    return action.item()


def update_policy(cfg, policy, optimizer):
    R = 0
    rewards = []
    policy_loss = []

    # Calculate reward from end -> start
    for r in policy.rewards[::-1]:
        R = r + cfg.gamma * R
        rewards.insert(0, R)

    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)
    # Max E(R(T)) = Max R( R_t * prob(a_t) )
    # == Min - E(R(T)) = Min - R( R_t * prob(a_t) )
    for R, log_prob in zip(rewards, policy.log_prob):
        policy_loss.append(-R * log_prob)

    # Basic idea: https://pytorch.org/docs/stable/optim.html
    # Explain backward() and step()
    # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step#:~:text=step()%20makes%20the%20optimizer,grad%20to%20update%20their%20values.&text=When%20you%20call%20loss.,and%20store%20them%20in%20parameter.

    # The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you
    # call backward() on the loss.

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_prob[:]


def main():
    cfg = arg_parser()
    env = gym.make('CartPole-v1')
    env.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    policy = Policy(obs_shape=env.observation_space.shape[0], act_shape=env.action_space.n)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    max_path_length = env.spec.max_episode_steps
    episode_rewards = []
    for episode_id in count(1):
        obs, episode_reward = env.reset(), 0

        for i in range(max_path_length):
            action = select_action(obs, policy)
            obs, reward, done, info = env.step(action)

            episode_reward += reward

            # Add reward of each step to policy, for policy updating
            policy.rewards.append(reward)

            if done:
                episode_rewards.append(episode_reward)
                break

        # Update policy each episode
        update_policy(cfg, policy, optimizer)

        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        if episode_id % cfg.log_interval == 0:
            print('episode_id:  ', episode_id, '    avg_reward:  ', avg_reward)
        if avg_reward >= env.spec.reward_threshold:
            print('Solved by using: ', episode_id, '    episodes!')
            break


if __name__ == '__main__':
    main()
