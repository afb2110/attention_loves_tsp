import torch


class MultiHeadAttentionLayer(torch.nn.Module):
    pass


class Encoder(torch.nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512):
        self.M = n_heads
        self.d_h = embed_dim
        self.N = n_layers
        self.d_x = node_dim
        self.normalization = normalization
        self.feed_forward_hidden = 512
        # self.first_embed = torch.nn.Linear(self.d_x, self.d_h, bias=True)  # Contains W and b  # TODO check initialization
        self.W_x = torch.nn.Parameter(torch.random(self.d_h, self.d_x))  # TODO check initialization
        self.W_x.requires_grad = True
        self.b = torch.nn.Parameter((torch.random(self.d_h)))  # TODO check initialization
        self.b.requires_grad = True
        self.layers = torch.nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(self.N)))  # Each attention layer contains an MHA and a FF

    def forward(self, x, mask=None):
        # h = self.first_embed(x)  # Creates the first embedding h from the data x
        h = torch.dot(self.W_x, x) + self.b

        h = self.layers(h)  # Goes through the N (MHA+FF) layers. We have the final node embeddings.

        h_graph = h.mean(1)  # embedding of the graph

        return h, h_graph


def new_attention_mechanism(node):  # TODO Details of attention mechanism
    return node


class Decoder(torch.nn.Module):
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
        self.W_q = torch.nn.Parameter(torch.random(self.d_h, self.d_h))
        self.W_k = torch.nn.Parameter(torch.random(self.d_h, self.d_x))
        self.W_v = torch.nn.Parameter(torch.random(self.d_h, self.d_x))
        self.context_node = [embedded_graph[1], torch.dot(self.W_v, embedded_graph[0][initial_node]),
                             torch.dot(self.W_v, embedded_graph[0][final_node])]

    def forward(self):
        not_visited = np.ones(self.n_nodes)
        for time_step in range(self.n_nodes):
            self.context_node = new_attention_mechanism(self.context_node)

