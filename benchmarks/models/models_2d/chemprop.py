# borrowed from https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop.py

import torch

from torch import nn
from torch_scatter import scatter_sum
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.data.data import size_repr
from torch_geometric.nn.resolver import activation_resolver

from models.models_2d.encoders import AtomEncoder, BondEncoder


class RevIndexedData(Data):
    def __init__(self, orig=None):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys:
                self[key] = orig[key]
            edge_index = self['edge_index']
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self['revedge_index'] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'revedge_index':
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return '{}({})'.format(cls, ', '.join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return '{}(\n{}\n)'.format(cls, ',\n'.join(info))


def transform_reversely_indexed_data(data):
    return RevIndexedData(data)


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class ChemProp(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, act='relu'):
        super(ChemProp, self).__init__()
        self.act = activation_resolver(act)
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)
        self.W1 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W3 = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.num_layers - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act(self.W3(z))

        # readout: pyg global pooling
        return global_mean_pool(node_attr, batch)
