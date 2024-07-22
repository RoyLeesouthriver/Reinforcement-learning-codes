import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Sarsa import CliffWalkingEnv,Sarsa,nstep_Sarsa
from Q_learning import QLearning

ncol = 12
nrow =4 
env = CliffWalkingEnv(ncol,nrow)
np.random.seed(0)
#randomseed会在以下方面产生影响：
#1.网络模型的初始化参数，小批次学习时对于样本的抽取逻辑，以及优化器的优化参数
#2.对于强化学习而言，由于实际是对依概率采样的状态进行学习，因此randomseed也会影响到环境的状态分布情况
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n_step = 5
agent = QLearning(ncol,nrow,epsilon,alpha,gamma)
# agent = nstep_Sarsa(n_step,ncol,nrow,epsilon,alpha,gamma)
num_episodes = 500

return_list = []
for i in range(10):
    #tqdm显示进度条
    with tqdm (total=int(num_episodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):#每个进度条对应的序列
            episode_return = 0 
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state , reward , done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward #回报计算不需要进行因子衰减
                # agent.update(state,action,reward,next_state,next_action,done)
                agent.update(state,action,reward,next_state)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if(i_episode +1) % 10 == 0:
                #每十条序列打印十条序列的平均回报
                pbar.set_postfix({'episode':'%d' % (num_episodes/10 * i + i_episode +1),'return':'%.3f' %np.mean(return_list[-10:])})
            pbar.update(1)

def print_agent(agent,env,action_meaning,disaster=[],end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****',end=' ')
            elif (i * env.ncol +j ) in end:
                print('EEEE',end='')
            else:
                a = agent.best_action( i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str,end=' ')
        print()

    

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))
plt.savefig("Q-learning_Cliff_walking.jpg")
action_meaning = ['^','v','<','>']
print('Q-learning算法最终收敛得到的策略为：')
print_agent(agent,env,action_meaning,list(range(37,47)),[47])