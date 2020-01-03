# TODO check if it is ok -- I did not check the implementation yet

from torch.utils.data import Dataset
import torch
import os
import pickle


class TSP(object):
    NAME = 'tsp'
    NODE_DIM = 2

    @staticmethod
    def get_costs(dataset, pi, log_p=False):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)


class TSP_greedy(TSP):
    @staticmethod
    def build_solution(dataset, pi_0):
        batch_size, graph_size, _ = dataset.size()
        dist = (dataset.transpose(1,2).repeat_interleave(1, graph_size, 1).transpose(1,2) - dataset.repeat(1, graph_size, 1)).norm(p=2, dim=2).view(1, graph_size, graph_size)
        M = dist.max()
        dist = dist + M * torch.eye(graph_size)
        
        tour = torch.zeros((batch_size, graph_size)).int()
        tour[:, 0] = pi_0
        current_node = pi_0
        
        i = 0
        while i < graph_size:
            next_node = dist[:, current_node, :].argmin(dim=1)
            tour[:, i] = next_node

            dist[:, next_node, :], dist[:, :, next_node] = M, M
            current_node = next_node
            i += 1
        
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(tour.size(1), out=tour.data.new()).view(1, -1).expand_as(tour) ==
                tour.data.sort(1)[0]
        ).all(), "Invalid tour"

        tour_one_hot = TSP_greedy.one_hot_solution()

        return tour, tour_one_hot

    @staticmethod
    def one_hot_solution(solution):#self):
        batch_size, graph_size = solution.size()
        tour_one_hot = torch.zeros((batch_size, graph_size, graph_size))
        for i in range(batch_size):
            tour_one_hot[i, torch.arange(graph_size), solution[i]] = 1
        #self.tour_one_hot = tour_one_hot

        return tour_one_hot


    @staticmethod
    def get_costs(pi, log_p):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size, graph_size) one-hot encoded permutations representing tours
        :return: (batch_size) cross-entropy of greedy solution
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        loss = -(pi*log_p).sum(dim=2).sum(dim=1)
        
        return loss


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in data[:num_samples]]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]