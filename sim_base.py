from base import BaseProcess

sim = BaseProcess(population=20, 
                  num_oss=2,
                  init_adopt_rate=0.2,
                  prob_contribute=0.2,
                  init_oss_max=20,
                  max_evol=20
                  )

if __name__=="__main__":
    sim.sim()