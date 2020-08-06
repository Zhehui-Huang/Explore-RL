# Paper: https://arxiv.org/pdf/1604.06778.pdf
# Compared with REINFORCE.py, we add baselines for REINFORCE algorithms
# Got the basic idea from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
# Solving Cartpole-v1 with 507 episodes
import os
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from algorithms.REINFORCE.config import arg_parser

EPS = np.finfo(np.float32).eps.item()
SavedACValue = namedtuple('SavedACValue', ['log_prob', 'critic_value'])


class Policy(nn.Module):
    def __init__(self, obs_shape=0, act_shape=0):
        super(Policy, self).__init__()
        assert obs_shape > 0 and act_shape > 0

        self.layer_1 = nn.Linear(obs_shape, 128)

        self.actor_head = nn.Linear(128, act_shape)
        self.critic_head = nn.Linear(128, 1)

        self.rewards = []
        self.saved_ac_value = []

    def forward(self, obs):
        x = self.layer_1(obs)
        x = F.relu(x)

        act_prob = self.actor_head(x)
        act_prob = F.softmax(act_prob, dim=-1)

        critic_value = self.critic_head(x)

        # act_prob: the probability of actions
        # critic_value: b(s_t), evaluate the value of state s_t
        return act_prob, critic_value


def select_action(obs, policy):
    # 1. change numpy,array to tensor, 2. ensure it is float, 3. unsqueeze, since nn.Module can only deal with
    # mini-batch, wich means even we only have one observation, we still need to unsqueeze it. change from CHW -> BCHW

    # obs = torch.from_numpy(obs).float().unsqueeze(dim=0)
    obs = torch.from_numpy(obs).float().unsqueeze(dim=0)

    act_prob, critic_value = policy(obs)

    # https://pytorch.org/docs/stable/distributions.html
    # Categorical has sample() and log_prob()
    m = Categorical(act_prob)
    action = m.sample()
    policy.saved_ac_value.append(SavedACValue(m.log_prob(action), critic_value))

    return action.item()


def update_policy(cfg, policy, optimizer):
    R = 0
    rewards = []
    policy_loss = []
    value_loss = []

    # Calculate reward from end -> start
    for r in policy.rewards[::-1]:
        R = r + cfg.gamma * R
        rewards.insert(0, R)

    # ** This is important, without this
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)
    # Max E(R(T)) = Max R( R_t * prob(a_t) )
    # == Min - E(R(T)) = Min - R( R_t * prob(a_t) )
    for R, (log_prob, critic_value) in zip(rewards, policy.saved_ac_value):
        advantage = R - critic_value.item()

        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(critic_value, torch.tensor([[R]])))

    # Basic idea: https://pytorch.org/docs/stable/optim.html
    # Explain backward() and step()
    # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step#:~:text=step()%20makes%20the%20optimizer,grad%20to%20update%20their%20values.&text=When%20you%20call%20loss.,and%20store%20them%20in%20parameter.

    # The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you
    # call backward() on the loss.

    optimizer.zero_grad()
    # policy_loss = torch.cat(policy_loss).sum()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_ac_value[:]


def enjoy(cfg):
    env = gym.make('CartPole-v1')
    env.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    path = os.path.join(os.getcwd(), 'tmp.pth')
    policy = torch.load(path)
    max_path_length = env.spec.max_episode_steps
    episode_rewards = []
    for episode_id in count(1):
        obs, episode_reward = env.reset(), 0

        for i in range(max_path_length):
            action = select_action(obs, policy)
            obs, reward, done, info = env.step(action)

            env.render()
            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                break

        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        if episode_id % cfg.log_interval == 0:
            print('episode_id:  ', episode_id, '    last_rew:   ', episode_reward, '    avg_reward:  ', avg_reward)
        else:
            print('episode_id:  ', episode_id, '    episode_reward:   ', episode_reward)


def main(cfg=None):
    framestep_count = 0
    writer = SummaryWriter('runs/test')
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
                framestep_count += i
                writer.add_scalar('reward',
                                  episode_reward,
                                  framestep_count)
                episode_rewards.append(episode_reward)
                break

        # Update policy each episode
        update_policy(cfg, policy, optimizer)

        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        writer.add_scalar('avg_reward', avg_reward, framestep_count)

        if episode_id % cfg.log_interval == 0:
            print('episode_id:  ', episode_id, '    avg_reward:  ', avg_reward)
        if avg_reward >= env.spec.reward_threshold:
            print('Solved by using: ', episode_id, '    episodes!')
            break

    path = os.path.join(os.getcwd(), 'tmp.pth')
    torch.save(policy, path)


if __name__ == '__main__':
    cfg = arg_parser()
    if cfg.train:
        main(cfg)
    else:
        enjoy(cfg)
