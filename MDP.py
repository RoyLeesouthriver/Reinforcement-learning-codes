#马尔可夫决策过程，是强化学习中的重要概念，
#不同于多臂老虎机，马尔可夫过程包含状态信息和状态之间的转移机制，
#对于实际问题中的强化学习而言，都往往需要将该问题转化为MDP问题，明确组成该问题的各个要素
import numpy as np
np.random.seed(0)
#定义状态概率转移矩阵P
P = [
    [0.9,0.1,0.0,0.0,0.0,0.0],
    [0.5,0.0,0.5,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.6,0.0,0.4],
    [0.0,0.0,0.0,0.0,0.3,0.7],
    [0.0,0.2,0.3,0.5,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,1.0],
]
P = np.array(P)
#奖励函数
rewards = [-1,-2,-2,10,1,0]
#折扣因子
gamma = 0.5
#给定一条序列，计算从某个起始状态到序列最后状态得到的回报
def compute_return(start_index,chain,gamma):
    G = 0
    for i in reversed(range(start_index,len(chain))):
        #从终止态开始进行序列添加，以此模拟gamma的多次幂
        G = gamma *G + rewards[chain[i] - 1]
    return G

def compute(P,rewards,gamma,states_num):
    '''利用贝尔曼方程的矩阵形式计算解析解，MRP状态数为num'''
    rewards = np.array(rewards).reshape((-1,1))#将rewards写为列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*P),rewards)
    return value

#状态序列s1-s2-s3-s6
chain = [1,2,3,6]
start_index = 0
G = compute_return(start_index,chain,gamma)
print("累计回报:%s" %G)
V = compute(P,rewards,gamma,6)
print("MRP中每个状态价值分别为\n",V)

S = ["s1","s2","s3","s4","s5"]#状态
A = ["保持s1","前往s1","前往s2","前往s3","前往s4","前往s5","概率前往"]#动作
#状态转移函数
P = {
    "s1-保持s1-s1":1.0,"s1-前往s2-s2":1.0,
    "s2-前往s1-s1":1.0,"s2-前往s3-s3":1.0,
    "s3-前往s4-s4":1.0,"s3-前往s5-s5":1.0,
    "s4-前往s5-s5":1.0,"s4-概率前往-s2":0.2,
    "s4-概率前往-s3":0.4,"s4-概率前往-s4":0.4,
}
#奖励函数
R = {
    "s1-保持s1":-1,"s1-前往s2":0,
    "s2-前往s1":-1,"s2-前往s3":-2,
    "s3-前往s4":-2,"s3-前往s5":0,
    "s4-前往s5":10,"s4-概率前往":1,
}
gamma = 0.5 #折扣因子
MDP = (S,A,P,R,gamma)
#策略1：随机选择
Pi_1 = {
    "s1-保持s1":0.5,"s1-前往s2":0.5,
    "s2-前往s1":0.5,"s2-前往s3":0.5,
    "s3-前往s4":0.5,"s3-前往s5":0.5,
    "s4-前往s5":0.5,"s4-概率前往":0.5,
}
#策略2
Pi_2 = {
    "s1-保持s1":0.6,"s1-前往s2":0.4,
    "s2-前往s1":0.3,"s2-前往s3":0.7,
    "s3-前往s4":0.5,"s3-前往s5":0.5,
    "s4-前往s5":0.1,"s4-概率前往":0.9,
}

def join(str1,str2):
    return str1+ '-' + str2

gamma = 0.5
#转化后的MRP状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5,0.5,0.0,0.0,0.0],
    [0.5,0.0,0.5,0.0,0.0],
    [0.0,0.0,0.0,0.5,0.5],
    [0.0,0.1,0.2,0.2,0.5],
    [0.0,0.0,0.0,0.0,1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5,-1.5,-1.0,5.5,0]
V = compute(P_from_mdp_to_mrp,R_from_mdp_to_mrp,gamma,5)
print("MDP中每个状态价值分别为:\n",V)
#蒙特卡洛采样

def sample(MDP,Pi,timestep_max,number):
    """采样函数，策略为Pi,限制最长时间步为timestamp_max,总采样序列数number"""
    S,A,P,R,gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] #随机选择除s5以外的状态s作为起点
        #当前状态为终止状态或时间步长太长时，一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand,temp = np.random.rand(),0
            #在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s,a_opt),0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s,a),0)
                    break
            rand,temp = np.random.rand(),0
            #根据状态转移概率求下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s,a),s_opt),0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s,a,r,s_next))#将元组放入序列内
            s = s_next #s_next变为当前状态，开始接下来的循环
        episodes.append(episode)
    return episodes

def MC(episodes,V,N,gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1,-1,-1):#序列从后向前计算
            (s,a,r,s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + ( G - V[s])/ N[s]


#采样五次，每次序列不超过最长的20步
episodes = sample(MDP,Pi_1,20,5)
print('第一条序列\n',episodes[0])
print('第二条序列\n',episodes[1])
print('第五条序列\n',episodes[4])
#采样长度
timestep_max = 20
#采样1000次
episodes = sample(MDP,Pi_1,timestep_max,10000)
gamma = 0.5
V = {"s1":0,"s2":0,"s3":0,"s4":0,"s5":0}
N = {"s1":0,"s2":0,"s3":0,"s4":0,"s5":0}
MC(episodes,V,N,gamma)
print("使用蒙塔卡罗方法计算的MDP状态价值为\n",V)