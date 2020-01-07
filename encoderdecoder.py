#!/usr/bin/env python

import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from graph_attention_layer import AttentionLayer, MultiHeadAttention, AttentionMechanismVaswani
from torch.autograd import Variable

from utils import log_values, maybe_cuda_model


class AttentionModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 decode_type="greedy",  # TODO change to sampling later
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8):
        super(AttentionModel, self).__init__()


        # self.mask_inner = mask_inner
        # self.mask_logits = mask_logits
        # self.hidden_dim = hidden_dim
        # self.n_encode_layers = n_encode_layers
        self.decode_type = decode_type
        # self.temp = 1.0  # If we add a temperature one day

        self.problem = problem

        self.embedder = Encoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            node_dim=problem.NODE_DIM,
            normalization=normalization
        )

        key_dim = embedding_dim // n_heads

        self.decoder = Decoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            key_dim=key_dim,
            tanh_clipping=tanh_clipping,
            decode_type=decode_type
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, eval_seq=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :param eval_seq: (batch_size, graph_size) sequence to score (supervised) or None for autoregressive
        :return:
        """

        embeddings, h_graph = self.embedder(input)

        log_p, pi = self.decoder(embeddings, h_graph, eval_seq)

        cost, mask = self.problem.get_costs(input, pi, log_p)

        return cost, log_p, pi, mask


class Encoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512):
        super(Encoder, self).__init__()

        self.d_x = node_dim  # Initial size of each node
        self.N = n_layers  # Number of MHA+FF layers

        # MHA Parameters
        self.M = n_heads
        self.d_h = embed_dim
        self.feed_forward_hidden = feed_forward_hidden
        self.normalization = normalization

        # Creating the initial linear projection with learned parameters
        self.first_embed = nn.Linear(self.d_x, self.d_h, bias=True)  # Contains W and b, uniform initialization

        # Creating the concatenation of the N MHA+FF layers
        self.layers = nn.Sequential(*(
            AttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(self.N)))  # Each attention layer contains an MHA and a FF

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Creating the first embedding h from the data x
        h = self.first_embed(x)  # TOCHECK We may need to change the dimention here

        # Going through the N (MHA+FF) layers. We have the final node embeddings.
        h = self.layers(h)

        # Embedding of the graph
        h_graph = h.mean(1)

        return h, h_graph


class Decoder(nn.Module):
    def __init__(self,
            n_heads,
            embed_dim,
            key_dim,
            tanh_clipping=10,
            decode_type="greedy"):
        super(Decoder, self).__init__()

        # Primitive attributes
        self.n_heads = n_heads
        self.input_dim = embed_dim  # dimension of h

        # Mechanism-dependent attributes
        self.key_dim = key_dim
        self.tanh_clipping = tanh_clipping
        self.decode_type = decode_type

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        n_heads_mha = 8
        embed_dim_mha = 3 * embed_dim
        self.MHA = MultiHeadAttention(n_heads_mha, embed_dim_mha, embed_dim_mha, AttentionMechanismVaswani,
                                      params_attention={'n_heads': n_heads_mha,
                                      'input_dim': embed_dim_mha,
                                      'key_dim': embed_dim_mha // n_heads,
                                      'value_dim': embed_dim_mha // n_heads}
        )
        self.W_Q = nn.Parameter(torch.randn((3 * embed_dim, key_dim)))
        self.W_K = nn.Parameter(torch.randn((embed_dim, key_dim)))

        # Create the placeholder for the first graph embedding
        std = 1. / math.sqrt(embed_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embed_dim))
        self.W_placeholder.data.uniform_(-std, std)

    def initialize_params(self, init_function):
        """Initialization."""
        init_function(self.W_Q)
        init_function(self.W_K)

    def forward(self, h, h_graph, eval_seq):

        batch_size, graph_size, input_dim = h.size()
        # h = h.view(batch_size, 1, graph_size, input_dim)

        h_graph = h_graph.view(batch_size, 1, input_dim)

        log_probs = []
        sequences = []
        # 0 --> 1 & 1 --> -inf
        # visited = Variable(h.data.new().byte().new(batch_size, graph_size).zero_())  # TOCHECK what does that mean
        visited = torch.zeros((batch_size, graph_size)) #, device=torch.device('cuda'))
        visited = visited.type(torch.bool)
        is_cuda = next(self.parameters()).is_cuda
        if is_cuda:
            visited = visited.cuda()

        for time_step in range(graph_size):

            # Computing compatibilities
            h_nodes = self._get_context_nodes(h, sequences)
            # q_nodes : (batch_size, graph_size, key_dim)
            h_nodes = h_nodes.view(batch_size, 1, 2 * input_dim)

            h_c = torch.cat([h_graph, h_nodes], dim=-1).view(batch_size, 1, 3 * input_dim)
            h_nodes = None  # Need to free memory, otherwise not enough RAM

            # Going through the mha layer
            # h_c = self.MHA(h_c)

            # q_graph : (batch_size, graph_size, key_dim)
            # Can be calculated outside the loop because constant graph embedding
            q = torch.matmul(h_c, self.W_Q)
            # h : (batch_size, graph_size, input_dim)
            #  W_k : (embed_dim, key_dim)
            # k : (batch_size, graph_size, key_dim)
            k = torch.matmul(h, self.W_K)
            k = k.squeeze()

            # compatibility : (batch_size, graph_size)
            compatibility = self.compat_factor * torch.matmul(q, k.transpose(1, 2))
            q, k = None, None
            compatibility = compatibility.squeeze()

            # From the logits compute the probabilities by clipping
            if self.tanh_clipping > 0:
                compatibility = torch.tanh(compatibility) * self.tanh_clipping

            compatibility[visited] = -math.inf  # or np.inf

            # attention : (batch_size, n_heads, graph_size)
            log_attention = F.log_softmax(compatibility, dim=-1)  # TODO maybe one day we can add a temperature
            attention = log_attention.exp()

            # Select the indices of the next nodes in the sequences (or evaluate eval_seq), result (batch_size) long
            selected = self._select_node(attention, visited) if eval_seq is None else eval_seq[:, time_step]

            log_probs.append(log_attention)
            sequences.append(selected)

            #  Updating mask
            visited = visited.clone().scatter_(1, selected.unsqueeze(-1), True)

        # Collected lists, return Tensor
        return torch.stack(log_probs, 1), torch.stack(sequences, 1)

    def _select_node(self, log_probs, mask):

        if self.decode_type == "greedy":
            _, selected = log_probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        # elif self.decode_type == "sampling":  # TODO for later
        #     selected = probs.multinomial(1).squeeze(1)
        #
        #     # Check if sampling went OK, can go wrong due to bug on GPU
        #     # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
        #     while mask.gather(1, selected.unsqueeze(-1)).data.any():
        #         print('Sampled bad values, resampling!')
        #         selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _get_context_nodes(self, embeddings, sequences):
        """
        Finds the context. If it is the first step (no sequences), we have it from placeholders. Else, we keep the first sequence, the last one.
        """
        # self.context_node = [embedded_graph[1], torch.dot(self.W_v, embedded_graph[0][initial_node]),
        #                      torch.dot(self.W_v, embedded_graph[0][final_node])]
        if len(sequences) == 0:
            batch_size = embeddings.size(0)
            # First step, use learned input symbol (placeholder)
            # No need to repeat, by adding dimension broadcasting will work
            return self.W_placeholder.unsqueeze(0).expand(batch_size, -1)
        else:
            batch_size = embeddings.size(0)
            # embeddings = embeddings.squeeze()
            # Return first and last node embeddings
            return torch.gather(
                embeddings,
                1,
                torch.stack((sequences[0], sequences[-1]), dim=1)
                .contiguous()
                .view(batch_size, 2, 1)
                .expand(batch_size, 2, embeddings.size(-1))
            ).view(batch_size, -1)  # View to have (batch_size, 2 * embed_dim)
            # return torch.gather(embeddings, 1, torch.stack((sequences[0], sequences[-1]), dim=1)).view(batch_size, -1)
