import random
import gym
import collections
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    '''经验回放池'''
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)#队列，采用FIFO机制
    
    def add(self,state,action,reward,next_state,done):
        #将数据加入buffer
        self.buffer.append((state,action,reward,next_state,done))
    
    def sample(self,batch_size):
        #从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
    
    def size(self):
        #目前buffer中的数据中的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    '''只有一层隐藏层的Q网络'''
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ConvolutionalQnet(torch.nn.Module):
    '''加入卷积层的Q网络'''
    def __init__(self,action_dim,in_channels=4):
        super(ConvolutionalQnet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,32,kernel_size=8,stride=4)
        self.conv2 = torch.nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = torch.nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.fc4 = torch.nn.Linear(7*7*64,512)
        self.head = torch.nn.Linear(512,action_dim)

    def forward(self,x):
        x = x/255
        x = F.relu(self.conv1(x))


class DQN:
    '''DQN算法'''
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device)#Q网络
        #目标网络
        self.target_q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        #Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon#贪婪策略参数
        self.target_update = target_update #网络参数更新频率
        self.count = 0 #计数器，记录更新次数
        self.device = device
    
    def take_action(self,state):#贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)#view(-1,1)转化为一列的张量
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        q_values = self.q_net(states).gather(1,actions) #Q值，从列中查询index
        #下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        #TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        #均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad() #默认梯度会累积，因此采用显示将梯度转化为0
        dqn_loss.backward()#反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) #更新目标网络
        self.count += 1


