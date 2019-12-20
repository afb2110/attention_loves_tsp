#!/usr/bin/env python

import os
import json
import pprint as pp

import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from options import get_options
from baselines import NoBaseline
from tsp import TSP as problem
from train import train_epoch, validate
from graph_attention_layer import AttentionLayer

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
            n_layers=self.n_encode_layers,
            node_dim=self.problem.NODE_DIM,
            normalization=normalization
        )

        key_dim = embedding_dim // n_heads

        self.decoder = Decoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            key_dim=key_dim,
            tanh_clipping=tanh_clipping
        )


    def forward(self, input, eval_seq=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :param eval_seq: (batch_size, graph_size) sequence to score (supervised) or None for autoregressive
        :return:
        """

        embeddings, h_graph = self.embedder(input)

        log_p, pi = self.decoder(embeddings, h_graph, eval_seq)

        cost, mask = self.problem.get_costs(input, pi)

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


if __name__ == "__main__":
    opts = get_options()

    # Set the random seed
    torch.manual_seed(0)

    # Load data from load_path
    load_data = {}
    if opts.load_path is not None:
        print('  [*] Loading data from {}'.format(opts.load_path))
        load_data = torch.load(opts.load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    # Initialize model
    model = maybe_cuda_model(AttentionModel(
            opts.embedding_dim,
            opts.hidden_dim,
            opts.problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization
        ),
        opts.use_cuda
    )


    # Overwrite model parameters by parameters to load
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    baseline = opts.baseline
    if baseline.isNone():
        baseline = NoBaseline()

    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': float(opts.lr)}])  # TODO: add parameters

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                opts
            )

class Decoder(nn.Module):
    def __init__(self,
            n_heads,
            embed_dim,
            key_dim,
            tanh_clipping=10):

        # Primitive attributes
        self.n_heads = n_heads
        self.input_dim = embed_dim  # dimension of h

        # Mechanism-dependent attributes
        self.key_dim = key_dim
        self.tanh_clipping = tanh_clipping

        self.compat_factor = 1 / math.sqrt(key_dim)  # faster than **(-0.5)

        self.W_Q_graph = nn.Parameter(torch.randn((n_heads, embed_dim, key_dim)))
        self.W_Q_nodes = nn.Parameter(torch.randn((n_heads, 2 * embed_dim, key_dim)))
        self.W_K = nn.Parameter(torch.randn((n_heads, embed_dim, key_dim)))


    def initialize_params(self, init_function):
        """Initialization."""
        init_function(self.W_Q)
        init_function(self.W_K)


    def forward(self, h, h_graph, eval_seq):


        batch_size, graph_size, input_dim = h.size()
        h = h.view(batch_size, 1, graph_size, input_dim)

        # q_graph : (batch_size, n_heads, graph_size, key_dim)
        q_graph = torch.matmul(h_graph, self.W_Q_graph)  # Can be calculated outside the loop because constant graph embedding

        log_probs = []
        sequences = []
        # 0 --> 1 & 1 --> -inf
        visited = np.zeros(graph_size)

        for time_step in range(graph_size):

            # Computing compatibilities

            h_nodes = self._get_context_nodes(self, h, sequences)
            # q_nodes : (batch_size, n_heads, graph_size, key_dim)
            q_nodes = torch.matmul(h_nodes, self.W_Q_nodes)
            # k : (batch_size, n_heads, graph_size, key_dim
            k = torch.matmul(h, self.W_K)
            q = q_graph + q_nodes

            # compatibility : (batch_size, n_heads, graph_size, graph_size)
            compatibility = self.compat_factor * torch.matmul(q, k.transpose(2, 3))

            # From the logits compute the probabilities by clipping
            if self.tanh_clipping > 0:
                compatibility = F.tanh(compatibility) * self.tanh_clipping

            mask_temp = np.clip((0.5 - visited)*np.inf, -np.inf, 1)  # TOCHECK -- there may be a better implementation since we make the mask ourselves
            mask_temp = torch.from_numpy(mask_temp)
            compatibility = compatibility * mask_temp

            # attention : (batch_size, n_heads, graph_size, graph_size)
            attention = F.softmax(compatibility, dim=-1)  # TODO maybe one day we can add a temperature

            idx_x, idx_y = np.where(visited != 1)  # TOCHECK - not needed normally
            attention[:, :, idx_x, idx_y] = 0

            # Select the indices of the next nodes in the sequences (or evaluate eval_seq), result (batch_size) long
            selected = self._select_node(log_probs, visited) if eval_seq is None else eval_seq[:, time_step]

            log_probs.append(attention.log())
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
            # First step, use learned input symbol (placeholder)
            # No need to repeat, by adding dimension broadcasting will work
            return self.W_placeholder.unsqueeze(0)
        else:
            batch_size = embeddings.size(0)
            # Return first and last node embeddings
            return torch.gather(
                embeddings,
                1,
                torch.stack((sequences[0], sequences[-1]), dim=1)
                .contiguous()
                .view(batch_size, 2, 1)
                .expand(batch_size, 2, embeddings.size(-1))
            ).view(batch_size, -1)  # View to have (batch_size, 2 * embed_dim)
