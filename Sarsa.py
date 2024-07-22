import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,ncol,nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0 #记录智能体的横坐标
        self.y = self.nrow -1 #记录智能体的纵坐标
    
    def step(self,action):#调用该函数进行位置的改变
        #定义动作为change[0]:up,change[1]:down,change[2]:left,change[3]:right,坐标系原点(0,0)定义在左上角
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        self.x = min(self.ncol -1,max(0,self.x + change[action][0]))
        #防止越界min，防止向下溢出max,取第一个数据，即x轴的变化
        self.y = min(self.nrow -1,max(0,self.y + change[action][1]))
        #防止越界min，防止向下溢出max,取第二个数据，即y轴的变化
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            #下一个动作已经是悬崖或者是目标
            done = True
            if self.x != self.ncol -1:
                reward = -100
        return next_state,reward ,done

    def reset(self):
        #回归初始状态
        self.x = 0
        self.y = self.nrow -1
        return self.y * self.ncol + self.x

class Sarsa:
    """Sarsa算法"""
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table = np.zeros([nrow*ncol,n_action])#初始化Q(s,a)表格
        self.n_action = n_action #动作个数
        self.alpha = alpha #学习率
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon #贪婪因子的随机选择动作概率
    
    def take_action(self,state):
        #选取下一步的具体动作
        if  np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self,state):
        #打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            #有相同价值的动作均进行记录
            if self.Q_table[state,i] == Q_max:
                a[i] = 1
        return a

    def update(self,s0,a0,r,s1,a1):
        td_error = r + self.gamma * self.Q_table[s1,a1] - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error
        
class nstep_Sarsa:
    """n步Sarsa算法"""
    def __init__(self,n,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table = np.zeros([nrow * ncol , n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n #采用n步Sarsa算法
        self.state_list = []
        self.action_list = []
        self.reward_list =[]
    
    def take_action(self,state):
        if  np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self,state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state,i] == Q_max:
                a[i] = 1
        return a
    
    def update(self,s0,a0,r,s1,a1,done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:
            #保存的数据可以进行n步更新
            G = self.Q_table[s1,a1] #获取Q(s(t+n),a(t+n))
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]
                #到达终止状态，即使长度不够，也要将其更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s,a] += self.alpha * (G - self.Q_table[s,a])
            s = self.state_list.pop(0) #将需要更新的状态动作从列表中删除，下次不必进行更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0) 
            #n步Sarsa的主要更新步骤
            self.Q_table[s,a] += self.alpha * (G-self.Q_table[s,a])
        if done:#到达终止状态开始下一条序列，将列表清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []