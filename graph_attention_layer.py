import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class AttentionMechanism(nn.Module):
    """docstring for AttentionMechanism."""

    def __init__(self):
        super(AttentionMechanism, self).__init__()


class AttentionMechanismVaswani(AttentionMechanism):

    def __init__(self,
                 n_heads,
                 input_dim,
                 key_dim,
                 value_dim):
        super(AttentionMechanismVaswani, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        self.W_Q = nn.Parameter(torch.tensor((n_heads, input_dim, key_dim)))
        self.W_K = nn.Parameter(torch.tensor((n_heads, input_dim, key_dim)))
        self.W_V = nn.Parameter(torch.tensor((n_heads, input_dim, value_dim)))

        # Initialization
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)


    def forward(self, h, mask=None):
        """
        h (batch_size, graph_size, input_dim)
        """
        
        q = torch.matmul(h, W_Q)
        k = torch.matmul(h, W_K)
        v = torch.matmul(h, W_V)

        # Transform anti-adjacency matrix into 
        mask = np.clip((0.5 - mask)*np.inf, -np.inf, 1))
        
        compatibility = self.compat_factor * torch.matmul(q, k.transpose(2, 3))
        compatibility = compatibility*mask
        



class MultiheadAttention(nn.Module):
    """docstring for AttentionLayer."""

    def __init__(self,
                 n_head,
                 input_dim,
                 embedding_dim,
                 attention_mechanism):
        super(AttentionLayer, self).__init__()

        self.n_head = n_head
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.attention_mechanism = attention_mechanism

        self.attention_mechanism()

        for param in self.parameters():
