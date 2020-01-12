from torch import nn
from nets.graph_encoder import Encoder


class CriticNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.init_embed = nn.Linear(input_dim, embedding_dim)
        self.encoder = Encoder(
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            node_dim=input_dim,
            normalization=encoder_normalization,
            feed_forward_hidden=512)

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        _, graph_embeddings = self.encoder(self.init_embed(inputs))
        return self.value_head(graph_embeddings)
