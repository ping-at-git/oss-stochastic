from base import BaseProcess
import numpy as np

class IIDProcess(BaseProcess):

    # def _init_pref(self, population, num_oss):
    #     """
    #     assign a 2D numpy array,
    #     here try uniform distribution
    #     """
    #     val = np.random.uniform(
    #         low=self.oss_value, high=100, size=(population, num_oss))
    #     self.pref = np.where(
    #         np.invert(self.state), val, 0)
        
    # def _sample_contribution(self, contributor):
    #     """
    #     return a numpy array containing increment per OSS
    #     """
    #     contribution = np.random.exponential(
    #         scale=1, size=self.state.shape) - 0.5
    #     return np.multiply(contribution, contributor).sum(axis=0)

    def _update_history(self):
        """get data"""
        adopt_count = self.history.get('adopt_count',[])
        adopt_count.append(self.state.sum(axis=0).tolist())
        self.history.update({'adopt_count': adopt_count})

        oss_value = self.history.get('oss_value',[])
        oss_value.append(self.oss_value)
        self.history.update({'oss_value': oss_value})

        # prob_adopt = self.history.get('prob_adopt', [])
        # prob_adopt.append(self.state.mean(axis=0))
        # self.history.update({'prob_adopt': prob_adopt})
