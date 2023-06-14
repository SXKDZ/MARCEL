import torch

from torch.nn import Linear, ReLU, Sequential, TransformerEncoderLayer, TransformerEncoder
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool


class SumPooling(torch.nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x, molecule_idx):
        x = global_add_pool(x, molecule_idx)
        return x


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, molecule_idx):
        x = global_mean_pool(x, molecule_idx)
        return x


class DeepSets(torch.nn.Module):
    def __init__(self, hidden_dim, reduce='mean'):
        super(DeepSets, self).__init__()
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.reduce = reduce

    def forward(self, x, molecule_idx):
        x = self.phi(x)
        x = scatter(x, molecule_idx, dim=0, reduce=self.reduce)
        x = self.rho(x)
        return x


class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attention = Linear(hidden_dim, hidden_dim)
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

    def forward(self, x, batch):
        x = self.phi(x)

        attention_scores = self.attention(x)
        dot_product = torch.matmul(attention_scores, attention_scores.transpose(1, 0))
        # attention_weights = scatter_softmax(dot_product, batch, dim=0)

        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        max_values = (dot_product * mask).max(dim=1, keepdim=True).values
        masked_dot_product = (dot_product - max_values) * mask
        attention_weights = masked_dot_product.exp() / (masked_dot_product.exp() * mask).sum(dim=1, keepdim=True)
        attention_weights = attention_weights * mask

        x_weighted = torch.matmul(attention_weights, x)
        x_aggregated = scatter(x_weighted, batch, dim=0, reduce='sum')

        x_encoded = self.rho(x_aggregated)
        return x_encoded


class TransformerPooling(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dim_feedforward, dropout):
        super(TransformerPooling, self).__init__()
        self.phi = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        transformer_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

    def forward(self, x, molecule_idx):
        x_transformed = self.phi(x)

        # Create a mask to prevent attention to padding tokens
        batch_size = len(molecule_idx.unique())
        max_set_size = molecule_idx.bincount().max()
        padding_mask = torch.zeros(batch_size, max_set_size, dtype=torch.bool)
        for batch_idx, count in enumerate(molecule_idx.bincount()):
            padding_mask[batch_idx, count:] = 1

        # Create a padded version of x_transformed to match the shape required by the Transformer encoder
        x_padded = torch.zeros(batch_size, max_set_size, x_transformed.size(1))
        batch_elements = (molecule_idx == torch.arange(batch_size).unsqueeze(1)).float()
        x_padded = torch.matmul(batch_elements, x_transformed)

        # Apply the Transformer encoder
        x_encoded_padded = self.transformer_encoder(x_padded, src_key_padding_mask=padding_mask)

        # Remove padding
        x_encoded = scatter(x_encoded_padded, molecule_idx, dim=0, reduce="sum")
        x_encoded = self.rho(x_encoded)

        return x_encoded


class Model4D(torch.nn.Module):
    def __init__(
            self, hidden_dim, out_dim,
            graph_model_factory, set_model_factory,
            device, unique_variables=1):
        super().__init__()
        self.graph_encoders = torch.nn.ModuleList(
            [graph_model_factory() for _ in range(unique_variables)])
        self.set_encoders = torch.nn.ModuleList(
            [set_model_factory() for _ in range(unique_variables)])
        self.linear = torch.nn.Linear(hidden_dim * unique_variables, out_dim)
        self.device = device

    def forward(self, batched_data, molecule_indices):
        outs = []
        for graph_encoder, set_encoder, data, molecule_index in zip(
                self.graph_encoders, self.set_encoders, batched_data, molecule_indices):
            data = data.to(self.device)
            z, pos, batch = data.x[:, 0], data.pos, data.batch
            out = graph_encoder(z, pos, batch)
            if graph_encoder.__class__.__name__ == 'LEFTNet':
                out = out[0]

            out = set_encoder(out, molecule_index)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.linear(outs).squeeze(-1)
        return outs


if __name__ == '__main__':
    model = SelfAttentionPooling(hidden_dim=16)
    x = torch.randn(11, 16)
    batch = torch.tensor([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 4])

    x_encoded = model(x, batch)

    x_encoded_manual_list = []
    batch_indices = batch.unique()

    for batch_idx in batch_indices:
        batch_elements = (batch == batch_idx).nonzero(as_tuple=True)[0]
        x_transformed_batch = model.phi(x[batch_elements])
        attention_scores_batch = model.attention(x_transformed_batch)
        dot_product_batch = torch.matmul(attention_scores_batch, attention_scores_batch.transpose(1, 0))

        softmax_batch = torch.exp(dot_product_batch) / torch.sum(torch.exp(dot_product_batch), dim=1, keepdim=True)
        x_weighted_batch = torch.matmul(softmax_batch, x_transformed_batch)
        x_aggregated_batch = x_weighted_batch.sum(dim=0, keepdim=True)
        x_encoded_manual_batch = model.rho(x_aggregated_batch)

        x_encoded_manual_list.append(x_encoded_manual_batch)

    x_encoded_manual = torch.cat(x_encoded_manual_list, dim=0)
    torch.testing.assert_close(x_encoded, x_encoded_manual, rtol=1e-5, atol=1e-5)
