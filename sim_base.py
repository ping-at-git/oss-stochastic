from oss_base import BaseProcess
from oss_competition import CompeteProcess

sim = BaseProcess(population=20, 
                  num_oss=2,
                  init_adopt_rate=0.2,
                  prob_contribute=0.2,
                  init_oss_max=20,
                  max_evol=20
                  )

tmp = CompeteProcess(population=20,
                        num_oss=3,
                        max_evol=100,
                        oss_discount= 1,
                        init_adopt_rate=0.2)


if __name__=="__main__":
    # sim.sim()
    tmp.sim()
