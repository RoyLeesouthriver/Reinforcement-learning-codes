##多臂老虎机，强化学习试错问题中最经典的范式
#定义多臂老虎机：有一个K根拉杆的老虎机，拉动每一根拉杆均对应一个关于奖励的概率分布R，每次拉动其中一根拉杆，就可以从该拉杆对应的奖励概率中获得一个奖励r。
#在未知奖励概率分布的前提下，操作T次拉杆期望获得最多的奖励r，因此需要探索拉杆的获奖概率和选择获奖最多的拉杆两个动作中进行权衡
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_results

class BernouliBandit:
    """伯努利多臂老虎机，K对应摇杆个数"""
    def __init__(self,k):
        self.probs = np.random.uniform(size=K)#随机生成K个0~1的数，表示拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)#获奖最大概率的拉杆
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self,k):
        #模拟玩家选择k号拉杆的动作
        if np.random.randn() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    """多臂老虎机算法的经典框架"""
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)#每根拉杆的尝试次数
        self.regret = 0.#当前步的累计懊悔
        self.actions = []#维护一个列表，记录每一步的动作和懊悔
        self.regrets = []
    
    def update_regret(self,k):
        #计算累计懊悔并进行保存，k对应本次动作选择的拉杆编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        #返回当前动作选择哪一根拉杆，由每个具体的策略进行实现
        raise NotImplementedError

    def run(self,num_steps):
        #运行一定次数，num_steps为总次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """epsilon贪婪算法，继承Solver类"""
    """
        贪婪算法在每次选择动作时，以1-ε的概率选择最大预测值的动作，而以ε的概率选择从K个动作中随机进行抽取
    """
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.k)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.k) #随机选择一根拉杆
        else:
            k = np.argmax(self.estimates) #选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k] + 1)*(r-self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    """时间衰减算法"""
    def __init__(self,bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.k)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        #取epsilon为1/t，对应一个随着时间逐渐衰减的权重系数，该方法使得模型能够逐渐收敛于最优方向
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0,self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k]+1) * (r-self.estimates[k])
        return k
    
#Upper Confidence Bound,其为基于不确定性的策略算法，霍夫丁不等式提出，UCB算法在每次拉杆前，先确定每根拉杆的期望奖励上界，使得拉动每一根拉杆的期望奖励只有较小的概率p超出上界，最终选择期望奖励上界最大的拉杆
class UCB(Solver):
    """UCB算法"""
    def __init__(self,bandit,coef,init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.coef = coef
    
    def run_one_step(self):
        self.total_count += 1
        #计算上置信界
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. /(self.counts[k] + 1 ) *(r - self.estimates[k])
        return k

#汤普森采样，假设拉动每根拉杆的奖励服从特定概率分布，按照拉动每根拉杆的期望奖励进行选择。由于计算所有拉杆的期望奖励代价较高，汤普森算法进行采样，对奖励概率分布进行一轮采样，选择最大奖励的动作。
#汤普森采样是计算所有拉杆最高奖励概率的蒙特卡罗方法，一般我们按照beta分布进行建模
class ThompsonSampling(Solver):
    """汤普森采样算法，继承Solver类"""
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a = np.ones(self.bandit.k)#列表，即每次拉杆奖励为1的次数
        self._b = np.ones(self.bandit.k)#列表，每次拉杆奖励为0的次数
    
    def run_one_step(self):
        samples = np.random.beta(self._a,self._b)#按照beta分布采样一组奖励样本
        k = np.argmax(samples) #选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r #更新Beta分布的第一个参数
        self._b[k] += (1-r) #更新第二个参数
        return k 

np.random.seed(4396)
K = 10
coef = 1
bandit_10_arm = BernouliBandit(K)
print("随机生成了%d臂老虎机" % K)
print("获奖概率最大的拉杆为%d号，获奖概率为%.4f" %(bandit_10_arm.best_idx,bandit_10_arm.best_prob))

epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm,epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为:',epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver],["EpsilonGreedy"])
#可以发现，在经历了开始的一小部分时间后，贪婪算法的累计懊悔呈线性增长，原因在于，随机拉杆的探索带来的懊悔值是固定的

decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
plot_results([decaying_epsilon_greedy_solver],["DecayingEpsilonGreedy"])
#随时间逐步衰减的贪婪算法能够使累计懊悔与时间步的关系变为次线性，相比于固定比例而言，能够进一步有效降低贪婪算法的累计懊悔

UCB_Solver = UCB(bandit_10_arm,coef)
UCB_Solver.run(5000)
print("上置信界算法的累计懊悔为：",UCB_Solver.regret)
plot_results([UCB_Solver],["UCB"])
#UCB算法主动进行上界的预测，推算出理论的最大上界，并沿着最大期望进行调整

thmopson_solver = ThompsonSampling(bandit_10_arm)
thmopson_solver.run(5000)
print('汤普森采样算法的累计懊悔：',thmopson_solver.regret)
plot_results([thmopson_solver],["ThompsonSampling"])
#结论，贪婪算法随时间呈线性懊悔累计，而剩下的几种算法均为次线性增长。