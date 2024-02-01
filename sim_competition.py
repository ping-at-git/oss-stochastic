from oss_competition import CompeteProcess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time 

POPULATION = 1e3


def sim_once():
    sim = CompeteProcess(population=POPULATION, 
                        num_oss=2,
                        init_adopt_rate=0.01,
                        prob_contribute=0.01,
                        init_oss_max=1,
                        oss_discount=0.99,
                        max_evol=100
                        )
    res = sim.sim()
    df = pd.DataFrame(res.get('adopt_count'))
 
    df.plot.line(c='black', alpha=0.1, legend=False, 
                 title='Adoption count over time')
    plt.ylim(0, POPULATION)
    plt.show()

    return df


def plot_3d(df):

    bin_edges = [*range(0, int(POPULATION)+1, int(POPULATION/50))]
    tmp = df.apply(lambda x: np.histogram(x, bins=bin_edges)[0],
                   axis=1)
    
    x, y = np.meshgrid(bin_edges[1:], df.index)
    data = np.vstack(tmp.values)

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(azim=-85, elev=15)
    surf = ax.plot_surface(x, y, data)
    ax.set_xlabel('Number of adoptions.')
    ax.set_ylabel('Evolution round.')
    ax.set_zlabel('Number of OSS.')
    plt.show()

    return None


if __name__=="__main__":

    _ = sim_once()

    # breakpoint()
   