import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, BatchNorm1d, Embedding
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.resolver import activation_resolver

from models.models_2d.encoders import AtomEncoder, BondEncoder


class GPS(torch.nn.Module):
    def __init__(self, hidden_dim, walk_length, num_heads, num_layers, act='relu', dropout=0.5):
        super().__init__()
        self.act = activation_resolver(act)

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.pe_lin = Linear(walk_length, hidden_dim)

        self.conv = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                self.act,
                Linear(hidden_dim, hidden_dim),
                self.act,
            )
            conv = GPSConv(hidden_dim, GINEConv(nn), heads=num_heads, attn_dropout=dropout)
            self.conv.append(conv)

    def forward(self, data):
        x, edge_index, edge_attr, batch, pe = data.x, data.edge_index, data.edge_attr, data.batch, data.pe

        x = self.atom_encoder(x) + self.pe_lin(pe)
        edge_attr = self.bond_encoder(edge_attr)
        for conv in self.conv:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return x


class GIN(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, act='relu', virtual_node=False):
        super().__init__()
        self.act = activation_resolver(act)

        self.conv = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.conv.append(
                GINEConv(Sequential(
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                    self.act,
                    Linear(hidden_dim, hidden_dim),
                    self.act)))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        self.lin = Linear(hidden_dim, hidden_dim)
        self.virtual_node = virtual_node

        if virtual_node:
            self.virtual_node_feature = Embedding(1, hidden_dim)
            self.virtual_node_feature.weight.data.fill_(0)
            self.virtual_edge_feature = Embedding(1, hidden_dim)
            self.virtual_edge_feature.weight.data.fill_(0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        if self.virtual_node:
            num_nodes = x.size(0)
            num_graphs = batch.max().item() + 1
            virtual_node_idx = torch.arange(num_nodes, num_nodes + num_graphs, device=x.device)
            virtual_edge_index = torch.stack([torch.arange(num_nodes, device=x.device), virtual_node_idx[batch]], dim=0)
            virtual_edge_attr = self.virtual_edge_feature(torch.zeros(num_nodes, device=x.device).long())

            edge_index = torch.cat([edge_index, virtual_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, virtual_edge_attr], dim=0)

            virtual_node_features = self.virtual_node_feature(torch.zeros(num_graphs, device=x.device).long())
            x = torch.cat([x, virtual_node_features], dim=0)

        for conv in self.conv:
            x = conv(x, edge_index, edge_attr)

        if self.virtual_node:
            x = x[:-num_graphs]

        x = global_add_pool(x, batch)
        x = self.lin(x)
        x = self.act(x)
        return x


class Model2D(torch.nn.Module):
    def __init__(self, model_factory, hidden_dim, out_dim, dropout, device, unique_variables=1):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [model_factory() for _ in range(unique_variables)])
        self.linear = torch.nn.Linear(hidden_dim * unique_variables, out_dim)
        self.dropout = dropout
        self.device = device

    def forward(self, batched_data):
        outs = []
        for model, data in zip(self.models, batched_data):
            data = data.to(self.device)
            out = model(data)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = F.dropout(outs, p=self.dropout, training=self.training)
        outs = self.linear(outs).squeeze(-1)
        return outs
