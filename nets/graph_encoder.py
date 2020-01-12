from torch import nn
from nets.graph_attention_layer import AttentionLayer, MultiHeadAttention, AttentionMechanismVaswani


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

        # Creating the concatenation of the N MHA+FF layers
        self.layers = nn.Sequential(*(
            AttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(self.N)))  # Each attention layer contains an MHA and a FF

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Going through the N (MHA+FF) layers. We have the final node embeddings.
        h = self.layers(x)

        # Embedding of the graph
        h_graph = h.mean(1)

        return h, h_graph