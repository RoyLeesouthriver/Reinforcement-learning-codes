import matplotlib.pyplot as plt

def plot_results(solvers,solver_names):
    """生成累计懊悔随着时间变化的图像，输入solvers是一个列表，列表中的每一个元素对应一种特定的策略，solver_names为一个列表，储存对应策略的名称"""
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title("%d-armed bandit" % solvers[0].bandit.k)
    plt.legend()
    plt.show()
    plt.savefig("results.jpg")