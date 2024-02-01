import numpy as np

class BaseProcess:

    oss_floor = 0.001
    oss_value = None
    oss_eval = None
    state = None
    pref = None
    history = {}

    def __init__(self, 
                 population=1e6, 
                 num_oss=10,
                 init_adopt_rate=1e-4,
                 prob_contribute=1e-3,
                 init_oss_max=1,
                 oss_discount=1,
                 max_evol=1e3) -> None:
        
        self.term = False
        self.max_evol = max_evol
        self.c_prob = prob_contribute
        self.discount = oss_discount

        self._init_oss_value(num_oss, init_oss_max)
        self._init_state(int(population), num_oss, init_adopt_rate)
        self._init_pref(int(population), num_oss, init_oss_max)
        self._init_oss_eval(int(population), num_oss)

    def _init_oss_value(self, num_oss, init_oss_max):
        """
        assign a 1D numpy array
        """
        self.oss_value = np.random.random_sample(num_oss) * init_oss_max

    def _init_state(self, population, num_oss, init_adopt_rate):
        """
        assign state as Numpy Boolean array(2D),
        random initial adoption
        """
        self.state = np.random.binomial(
            1, init_adopt_rate, size=(population, num_oss)
        ).astype(np.bool_)

    def _init_pref(self, population, num_oss, init_oss_max):
        """
        assign a 2D numpy array,
        min=1, max=10, distribution can change
        """
        val = np.random.normal(
            loc=5, scale=2, size=(population, num_oss)
            ).clip(min=self.oss_floor, max=init_oss_max*10)
        self.pref = np.where(
            np.invert(self.state), self.oss_value+val, 0.01)

    def _init_oss_eval(self, population, num_oss):
        """
        assign a 2D numpy array,
        add noise to the objective value
        """
        noise = np.random.uniform(
            low=0, high=1, size=(population, num_oss))
        self.oss_eval = np.where(
            self.state, self.pref+noise, self.oss_value-noise)

    def _update_eval(self):
        param = 0.7
        self.oss_eval = np.add(self.oss_eval * param, 
                               self.oss_value * (1-param))

    def _switch_adoption(self):
        """TO override"""
        pass

    def _new_adoption(self):
        new_adoption = self.pref <= self.oss_eval
        self.state = np.add(self.state, new_adoption)

    def _evol_adopt(self):
        self._update_eval()
        self._switch_adoption()
        self._new_adoption()

    def _sample_contributor(self):
        """
        return a Boolean array
        """
        prob = np.random.random_sample(size=self.state.shape)
        c_bool = prob <= self.c_prob
        contributor = np.logical_and(self.state, c_bool)
        return contributor

    def _sample_contribution(self, contributor):
        """
        return a numpy array containing increment per OSS
        """
        contribution = np.random.exponential(
            scale=0.1, size=self.state.shape)
        return np.multiply(contribution, contributor).sum(axis=0)

    def _update_oss(self, val):
        self.oss_value = np.add(
            self.oss_value * self.discount, val)
    
    def _evol_contribute(self):
        contributor = self._sample_contributor()
        val = self._sample_contribution(contributor)
        self._update_oss(val)

    def _update_history(self):
        """TO override"""
        pass

    def _evolve(self):
        self._evol_adopt()
        self._evol_contribute()
        self._update_history()

    def sim(self):
        iter_idx = 0
        self._update_history()
        while ((not self.term) and (iter_idx<self.max_evol)):
            self._evolve()
            iter_idx += 1
        return self.history