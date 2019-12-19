# TODO for now, no baseline

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from scipy.stats import ttest_rel
import copy
from train import rollout

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class GreedyBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()
        self.v = None

    def eval(self, x, c):
        #TODO if we want greedy baseline

        # if self.v is None:
        #     v = c.mean()
        # else:
        #     v = self.beta * self.v + (1. - self.beta) * c.mean()
        #
        # self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']