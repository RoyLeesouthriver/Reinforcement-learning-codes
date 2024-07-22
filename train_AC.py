import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import Actor_Critic as AC

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128 
gamma = 0.98
device = torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = AC.ActorCritic(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device)

return_list = rl_utils.train_on_policy_agent(env,agent,num_episodes)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()
plt.savefig('Actor-Critic_CartPolev0.jpg')

mv_return = rl_utils.moving_average(return_list,9)
plt.plot(episodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.savefig('Smoothed_Actor_Critic_CartPolev0.jpg')