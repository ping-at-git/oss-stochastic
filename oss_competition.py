from oss_base import BaseProcess
import numpy as np


class CompeteProcess(BaseProcess):
    """
    Modify from base to incorporate the competition
    between OSS;
    Parameters with a different meaning:
    - init_adoption_rate: x percent of the population
                          adopted one competing OSS;
    """

    oss_abdprcs = None
    oss_swtprcs = None
    oss_exppref = None
    gamma = None

    def _init_state(self, population, num_oss, init_adopt_rate):
        cand = np.random.binomial(
            1, init_adopt_rate, population)
        oss_idx = np.random.choice(
            [*range(num_oss)], size=population)

        state = np.zeros((population, num_oss), dtype=np.bool_)
        for idx in range(population):
            if cand[idx] == 1:
                state[idx, oss_idx[idx]] = True
        self.state = state

    def _init_pref(self, population, num_oss, init_oss_max):
        cand = np.random.normal(
            loc=5, scale=2, size=(population, 1)
            ).clip(min=self.oss_floor, max=init_oss_max*10)
        mdf = np.any(self.state, axis=1).reshape(-1, 1)
        cand = np.where(mdf, self.oss_floor, cand)
        self.pref = np.tile(cand, (1, num_oss))

    def _init_oss_eval(self, population, num_oss):
        self.oss_eval = np.random.exponential(
            self.oss_value, size=(population, num_oss))

    def _check_exist_adopter(self):
        return np.any(self.state, axis=1)
    
    def _cal_adoption_prob(self, current_adopter):
        if self.oss_exppref is None:
            self.oss_exppref = np.exp(self.pref)

        nmr = np.exp(self.oss_eval)
        dmn_cmp = np.where(nmr > self.oss_exppref, 
                           nmr, 
                           self.oss_exppref)
        dmn = dmn_cmp.sum(axis=1).reshape(-1, 1)
        dmn = np.tile(dmn, (1, nmr.shape[-1]))

        prob = np.divide(nmr, dmn)
        mdf = np.tile(current_adopter.reshape(-1,1),
                      (1, nmr.shape[-1]))
        return np.multiply(prob, np.invert(mdf))
    
    def _sample_new_adoption(self, prob):
        pop,oss = prob.shape
        sample_prob = np.tile(
            np.random.random(pop).reshape(-1,1), oss)
        cum_prob = np.cumsum(prob, axis=1)
        cum_prob = np.insert(
            cum_prob, [0], np.zeros((pop,1)), axis=1)
        return np.where(
            (sample_prob>cum_prob[:,:oss]) & (sample_prob<cum_prob[:,1:]),
            True, False
        )
    
    def _update_new_adoption(self, new_adoption):
        new = np.logical_and(
            new_adoption, np.invert(self.oss_abdprcs))
        self.state = np.add(self.state, new)

    def _new_adoption(self):
        cand = self._check_exist_adopter()
        prob = self._cal_adoption_prob(cand)
        new_adoption = self._sample_new_adoption(prob)
        self._update_new_adoption(new_adoption)

    def _update_abdprcs(self):
        beta, eps = 0.5, 1e-5

        val_abandon = np.subtract(
            self.oss_eval, beta*self.pref)
        val_abandon = np.multiply(self.state, val_abandon)
        new_abandon = np.add(val_abandon, eps) <= 0

        if self.oss_abdprcs is None:
            self.oss_abdprcs = new_abandon
        else:
            self.oss_abdprcs = np.logical_and(
                self.oss_abdprcs, new_abandon
            )
        return new_abandon

    def _abandon_adoption(self):
        new_abandon = self._update_abdprcs()
        self.state = np.logical_and(
            self.state, np.logical_not(new_abandon))
        
    def _switch_adoption(self):
        pass

    def _evol_adopt(self):
        self._update_eval()
        self._abandon_adoption()
        self._switch_adoption()
        self._new_adoption()
        # breakpoint()

    def _update_history(self):
        adopt_count = self.history.get('adopt_count',[])
        adopt_count.append(self.state.sum(axis=0).tolist())
        self.history.update({'adopt_count': adopt_count})

        oss_value = self.history.get('oss_value',[])
        oss_value.append(self.oss_value)
        self.history.update({'oss_value': oss_value})
