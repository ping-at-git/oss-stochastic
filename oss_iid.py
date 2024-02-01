from oss_base import BaseProcess
import numpy as np

class IIDProcess(BaseProcess):
    """
    Newly added options below:
    - uniformly distributed preference instead of normal distribution;
    - negative contribution, by the exponential family;
    - reverse adoption, i.e. abandonment;
    """

    # def _init_pref(self, population, num_oss):
    #     """
    #     assign a 2D numpy array,
    #     here try uniform distribution
    #     """
    #     val = np.random.uniform(
    #         low=50, high=51, size=(population, num_oss))
    #     self.pref = np.where(
    #         np.invert(self.state), val, 0)
        
    # def _sample_contribution(self, contributor):
    #     """
    #     return a numpy array containing increment per OSS
    #     """
    #     contribution = np.random.exponential(
    #         scale=1, size=self.state.shape) - 0.5
    #     return np.multiply(contribution, contributor).sum(axis=0)

    def _switch_adoption(self):
        """reverse adoption"""
        rv_adoption = 0.85*self.pref >= self.oss_eval
        rv_adoption = np.logical_and(rv_adoption, self.state)
        self.state = np.where(rv_adoption, False, self.state)

    def _update_history(self):
        """get data"""
        adopt_count = self.history.get('adopt_count',[])
        adopt_count.append(self.state.sum(axis=0).tolist())
        self.history.update({'adopt_count': adopt_count})

        oss_value = self.history.get('oss_value',[])
        oss_value.append(self.oss_value)
        self.history.update({'oss_value': oss_value})
