from oss_competition import CompeteProcess
import pandas as pd
import matplotlib.pyplot as plt

POPULATION = 1e3
SIM_COUNT = 20


def sim_once():
    sim = CompeteProcess(population=POPULATION, 
                        num_oss=2,
                        init_adopt_rate=0.0,
                        prob_contribute=0.005,
                        init_oss_max=[2,2],  #int [1,10] or list
                        oss_discount=1,
                        max_evol=500
                        )
    res = sim.sim()
    df = pd.DataFrame(res.get('adopt_count'))
    return df


def plot_laststate(df):
    """only with 2 OSS for now"""
    df.divide(POPULATION).T.plot.scatter(
        x=0, y=1, s=30, c='black', alpha=0.05,
        xlim=(0,1), ylim=(0,1)
    )
    plt.show()


if __name__=="__main__":

    res = pd.DataFrame()
    for idx in range(SIM_COUNT):
        tmp = sim_once()
        res = pd.concat([res, tmp.iloc[-1]], axis=1)

        tmp.plot.line(c='black', alpha=0.1, legend=False, 
                     title='Adoption count over time')
        plt.ylim(0, POPULATION)
        plt.show()
        
    plot_laststate(res)
    # breakpoint()
   