#!/usr/bin/env python

import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple

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
            decode_type=decode_type,
            problem=problem
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

        log_p, pi = self.decoder(embeddings, h_graph, eval_seq, input)

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
            decode_type="greedy",
            problem=None):
        super(Decoder, self).__init__()

        self.problem = problem
        self.mask_inner = True
        self.mask_logits = True
        self.temp = 1.0

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

        embedding_dim = embed_dim
        step_context_dim = 2 * embedding_dim
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )




    def initialize_params(self, init_function):
        """Initialization."""
        init_function(self.W_Q)
        init_function(self.W_K)

    def forward(self, h, h_graph, eval_seq, input):
        #
        # batch_size, graph_size, input_dim = h.size()
        # # h = h.view(batch_size, 1, graph_size, input_dim)
        #
        # h_graph = h_graph.view(batch_size, 1, input_dim)
        #
        # log_probs = []
        # sequences = []
        # # 0 --> 1 & 1 --> -inf
        # # visited = Variable(h.data.new().byte().new(batch_size, graph_size).zero_())  # TOCHECK what does that mean
        # visited = torch.zeros((batch_size, graph_size)) #, device=torch.device('cuda'))
        # visited = visited.type(torch.bool)
        # is_cuda = next(self.parameters()).is_cuda
        # if is_cuda:
        #     visited = visited.cuda()
        #
        # for time_step in range(graph_size):
        #
        #     # Computing compatibilities
        #     h_nodes = self._get_context_nodes(h, sequences)
        #     # q_nodes : (batch_size, graph_size, key_dim)
        #     h_nodes = h_nodes.view(batch_size, 1, 2 * input_dim)
        #
        #     h_c = torch.cat([h_graph, h_nodes], dim=-1).view(batch_size, 1, 3 * input_dim)
        #     h_nodes = None  # Need to free memory, otherwise not enough RAM
        #
        #     # Going through the mha layer
        #     # h_c = self.MHA(h_c)
        #
        #     # q_graph : (batch_size, graph_size, key_dim)
        #     # Can be calculated outside the loop because constant graph embedding
        #     q = torch.matmul(h_c, self.W_Q)
        #     # h : (batch_size, graph_size, input_dim)
        #     #  W_k : (embed_dim, key_dim)
        #     # k : (batch_size, graph_size, key_dim)
        #     k = torch.matmul(h, self.W_K)
        #     k = k.squeeze()
        #
        #     # compatibility : (batch_size, graph_size)
        #     compatibility = self.compat_factor * torch.matmul(q, k.transpose(1, 2))
        #     q, k = None, None
        #     compatibility = compatibility.squeeze()
        #
        #     # From the logits compute the probabilities by clipping
        #     if self.tanh_clipping > 0:
        #         compatibility = torch.tanh(compatibility) * self.tanh_clipping
        #
        #     compatibility[visited] = -math.inf  # or np.inf
        #
        #     # attention : (batch_size, n_heads, graph_size)
        #     log_attention = F.log_softmax(compatibility, dim=-1)  # TODO maybe one day we can add a temperature
        #     attention = log_attention.exp()
        #
        #     # Select the indices of the next nodes in the sequences (or evaluate eval_seq), result (batch_size) long
        #     selected = self._select_node(attention, visited) if eval_seq is None else eval_seq[:, time_step]
        #
        #     log_probs.append(log_attention)
        #     sequences.append(selected)
        #
        #     #  Updating mask
        #     visited = visited.clone().scatter_(1, selected.unsqueeze(-1), True)
        #
        # # Collected lists, return Tensor
        # return torch.stack(log_probs, 1), torch.stack(sequences, 1)
        batch_size, graph_size, input_dim = h.size()
        self.shrink_size = None

        embeddings = h

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

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



    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)


    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2,
                                                                                   embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)
