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
        self.out_dim = key_dim
        self.value_dim = value_dim

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        self.W_Q = nn.Parameter(torch.tensor((n_heads, input_dim, key_dim)))
        self.W_K = nn.Parameter(torch.tensor((n_heads, input_dim, key_dim)))
        self.W_V = nn.Parameter(torch.tensor((n_heads, input_dim, value_dim)))


        # Initialization
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)


    def forward(self, h, mask=None):
        """
        h (batch_size, graph_size, input_dim)
        """
        batch_size, graph_size, input_dim = h.size()
        h_reshaped = h.view(batch_size, 1, graph_size, input_dim)
        q = torch.matmul(h_reshaped, W_Q)
        k = torch.matmul(h_reshaped, W_K)
        v = torch.matmul(h_reshaped, W_V)

        compatibility = self.compat_factor * torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            # Transform anti-adjacency matrix into a compatibility mask:
            # 0 --> 1 & 1 --> -inf
            mask = np.clip((0.5 - mask)*np.inf, -np.inf, 1)
            mask = torch.from_numpy(mask)
            
            compatibility = compatibility * mask

        attention = F.softmax(compatibility, dim=-1)
    
        if mask is not None:
            idx_x, idx_y = np.where(mask != 1)
            attention[:, :, idx_x, idx_y] = 0
        
        h_p = torch.matmul(attention, v)

        return h_p
        



class MultiheadAttention(nn.Module):
    """docstring for AttentionLayer."""

    def __init__(self,
                 n_head,
                 input_dim,
                 message_dim,
                 embed_dim,
                 attention_mechanism,
                 params_attention):
        super(MultiheadAttention
        self.attention_mechanism = self.attention_mechanism(**params_attention)

        self.n_head = n_head
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.embed_dim = embed_dim

        # TODO: find a way to unify self.attention_mecchanism.out_dim, message_dim
        self.W_O = nn.Parameter(torch.tensor((n_head, self.attention_mechanism.out_dim, embed_dim)))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)
    
    def forward(self, h, mask=None):
        h_p = self.attention_mechanism(h)

        out = torch.mm(
            h_p.permute(1, 2, 0, 3).contiguous().view(-1, self.n_head * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out