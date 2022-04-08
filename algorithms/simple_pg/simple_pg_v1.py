import argparse
import gym
from gym.spaces import Discrete, Box

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epoches=50, batch_size=5000, render=False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), 'This example only works for envs with continuous state spaces.'
    assert isinstance(env.action_space, Discrete), 'This example only works for envs with discrete action spaces.'

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n


    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)

    args = parser.parse_args()

    print('Using simplest formulation of policy gradient.')
    train(env_name=args.env_name, render=args.render, lr=args.lr)