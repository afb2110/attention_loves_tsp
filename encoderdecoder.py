import torch
from torch import nn
from graph_attention_layer import AttentionMechanism

class MultiHeadAttentionLayer(nn.Module):
    pass


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

        # Creating the first embedding h from the data x
        h = self.first_embed(x)

        # Going through the N (MHA+FF) layers. We have the final node embeddings.
        h = self.layers(h)

        # Embedding of the graph
        h_graph = h.mean(1)

        return h, h_graph


def new_attention_mechanism(node):  # TODO Details of attention mechanism
    return node


class Decoder(nn.Module):
    def __init__(self, embedded_graph, initial_node, final_node,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None):
        self.M = n_heads
        self.d_h = embed_dim
        self.N = n_layers
        self.n_nodes = len(x)
        # self.time_step = 0
        self.W_q = nn.Parameter(torch.random(self.d_h, self.d_h))
        self.W_k = nn.Parameter(torch.random(self.d_h, self.d_x))
        self.W_v = nn.Parameter(torch.random(self.d_h, self.d_x))
        self.context_node = [embedded_graph[1], torch.dot(self.W_v, embedded_graph[0][initial_node]),
                             torch.dot(self.W_v, embedded_graph[0][final_node])]

    def forward(self):
        not_visited = np.ones(self.n_nodes)
        for time_step in range(self.n_nodes):
            self.context_node = new_attention_mechanism(self.context_node)

