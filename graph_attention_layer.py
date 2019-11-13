import torch
from torch import nn

class AttentionMechanism(nn.Module):
    """docstring for AttentionMechanism."""

    def __init__(self, arg):
        super(AttentionMechanism, self).__init__()
        self.arg = arg



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
