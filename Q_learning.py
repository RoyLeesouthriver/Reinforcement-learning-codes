import numpy as np

class QLearning:
    """Q-learning算法"""
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table = np.zeros([nrow * ncol,n_action]) #初始化表格
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon #贪婪策略中的参数

    def take_action(self,state):#选取下一步动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self,state):#打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
             if self.Q_table[state,i] == Q_max:
                 a[i] = 1
        return a
    
    def update(self,s0,a0,r,s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0,a0]
        self.Q_table[s0,a0] += self.alpha * td_error

