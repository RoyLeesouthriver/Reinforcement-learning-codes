import random
import gym
import collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import DQN

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
# env.seed(0) 高版本已废除
env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = DQN.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN.DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)
return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d' % i ) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            state = env.reset(seed=0)[0]
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, truncated , _= env.step(action)
                done = done or truncated
                replay_buffer.add(state,action,reward,next_state,done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s,b_a,b_r,b_ns,b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states':b_s,'actions':b_a,'next_states':b_ns,'rewards':b_r,'dones':b_d}
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if(i_episode+1) %10 ==0:
                pbar.set_postfix({'episode':'%d' %(num_episodes/10 * i + i_episode + 1),'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('DQN_CartPole.jpg')
mv_return = rl_utils.moving_average(return_list,9)
plt.plot(episodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.savefig('DQN_mv_avg.jpg')

