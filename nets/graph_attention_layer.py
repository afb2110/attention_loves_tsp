import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AttentionMechanism(nn.Module):
    """Actual Attention Mechanism to use."""

    def __init__(self,
                 n_heads,
                 input_dim,
                 out_dim):
        super(AttentionMechanism, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.out_dim = out_dim

    def forward(self, *inputs):
        raise NotImplementedError


class AttentionMechanismVaswani(AttentionMechanism):
    """Attention Mechanism proposed by Vaswani."""

    def __init__(self,
                 n_heads,
                 input_dim,
                 key_dim,
                 value_dim):
        super(AttentionMechanismVaswani, self).__init__(n_heads, input_dim, value_dim)

        # Primitive attributes
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.out_dim = value_dim

        # Mechanism-dependent attributes
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        self.W_Q = nn.Parameter(torch.randn((n_heads, input_dim, key_dim)))
        self.W_K = nn.Parameter(torch.randn((n_heads, input_dim, key_dim)))
        self.W_V = nn.Parameter(torch.randn((n_heads, input_dim, value_dim)))


    def forward(self, h, mask=None):
        """
        h (batch_size, graph_size, input_dim)
        mask must be an anti-adjacency matrix.
        """
        batch_size, graph_size, input_dim = h.size()
        h = h.view(batch_size, 1, graph_size, input_dim)

        # q : (batch_size, n_heads, graph_size, key_dim)
        q = torch.matmul(h, self.W_Q)
        # k : (batch_size, n_heads, graph_size, key_dim
        k = torch.matmul(h, self.W_K)
        # v : (batch_size, n_heads, graph_size, value_dim)
        v = torch.matmul(h, self.W_V)

        # compatibility : (batch_size, n_heads, graph_size, graph_size)
        compatibility = self.compat_factor * torch.matmul(q, k.transpose(2, 3))

        if mask is not None:
            # Transform anti-adjacency matrix into a compatibility mask:
            # 0 --> 1 & 1 --> -inf
            mask = np.clip((0.5 - mask)*np.inf, -np.inf, 1)
            mask = torch.from_numpy(mask)

            compatibility = compatibility * mask

        # attention : (batch_size, n_heads, graph_size, graph_size)
        attention = F.softmax(compatibility, dim=-1)

        if mask is not None:
            idx_x, idx_y = np.where(mask != 1)
            attention[:, :, idx_x, idx_y] = 0

        # h_p : (batch_size, n_heads, graph_size, value_dim)
        h_p = torch.matmul(attention, v)

        return h_p


class AttentionMechanismVelickovic(AttentionMechanism):

    def __init__(self,
                 n_heads,
                 input_dim,
                 key_dim,
                 value_dim):
        super(AttentionMechanismVelickovic, self).__init__(n_heads, input_dim, value_dim)

        print("WARNING: AttentionMechanismVelickovic is not finished yet. Please use AttentionMechanismVaswani.")
        breakpoint()

        # TODO finish

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        self.W = nn.Parameter(torch.randn((n_heads, input_dim, key_dim)))
        self.A = nn.Parameter(torch.randn((n_heads, key_dim)))

        self.W_V = nn.Parameter(torch.randn((n_heads, input_dim, value_dim)))

        # Initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, h, neighbours, mask=None):
        """
        h (batch_size, graph_size, input_dim)
        """

        q = torch.matmul(h, self.W_Q)
        k = torch.matmul(h, self.W_K)
        v = torch.matmul(h, self.W_V)

        compatibility = self.compat_factor * torch.matmul(q, k.transpose(2, 3))


        if mask is not None:
            # Transform anti-adjacency matrix into a compatibility mask:
            # 0 --> 1 & 1 --> -inf
            mask = np.clip((0.5 - mask) * np.inf, -np.inf, 1)
            mask = torch.from_numpy(mask)

            compatibility = compatibility * mask

        attention = F.softmax(compatibility, dim=-1)

        if mask is not None:
            idx_x, idx_y = np.where(mask != 1)
            attention[:, :, idx_x, idx_y] = 0

        h_p = torch.matmul(attention, v)

        return h_p


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self,
                 n_heads,
                 input_dim,
                 embed_dim,
                 attention_mechanism,
                 params_attention):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.attention_mechanism = attention_mechanism(**params_attention)
        self.message_dim = self.attention_mechanism.out_dim

        self.W_O = nn.Parameter(torch.randn((self.n_heads, self.message_dim, self.embed_dim)))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

        self.attention_mechanism.initialize_params(nn.init.xavier_uniform_)

    def forward(self, h, mask=None):
        batch_size, graph_size, _ = h.size()

        # h_p : (batch_size, n_heads, graph_size, message_dim)
        h_p = self.attention_mechanism(h, mask=mask)

        # out : (batch_size, n_heads, graph_size, embed_dim)
        out = torch.matmul(h_p, self.W_O)

        # out : (batch_size, graph_size, embed_dim)
        #out = torch.matmul(out.permute(0, 2, 3, 1), torch.ones((batch_size, graph_size, self.n_heads, 1), device=torch.device('cuda'))).view(batch_size, graph_size, self.embed_dim)
        out = out.permute(0, 2, 3, 1).sum(dim=3).view(batch_size, graph_size, self.embed_dim)

        return out

""" n_heads = 3
input_dim = 4
key_dim = 5
value_dim = 6
batch_size = 1
graph_size = 7

AMV = AttentionMechanismVaswani(n_heads, input_dim, key_dim, value_dim)
h = torch.zeros(batch_size, graph_size, input_dim)
y = AMV(h) """


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class AttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(AttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    attention_mechanism=AttentionMechanismVaswani,
                    params_attention={'n_heads': n_heads,
                                      'input_dim': embed_dim,
                                      'key_dim': embed_dim // n_heads,
                                      'value_dim': embed_dim // n_heads}
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
