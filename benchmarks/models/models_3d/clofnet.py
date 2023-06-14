import math
import torch
import torch.nn.functional as F

from math import pi
from torch import nn, Tensor
from torch.nn import Embedding

from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import PairTensor


def swish(x):
    return x * torch.sigmoid(x)


# radial basis function to embed distances
# borrowed from SphereNet
class RBFEmb(nn.Module):
    """
    modified: delete cutoff with r
    """
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * \
                  (torch.cos(dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(-self.betas * torch.square((torch.exp(-dist) - self.means)))


class NeighborEmb(MessagePassing):
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(
            hid_dim, elementwise_affine=False)

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s += s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class TransformerConv(MessagePassing):
    def __init__(
            self, in_channels, out_channels, concat=True, beta=False,
            dropout=0., edge_dim=None, bias=True, root_weight=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Linear(in_channels[0], out_channels)
        self.lin_query = nn.Linear(in_channels[1], out_channels)
        self.lin_value = nn.Sequential(
            nn.Linear(in_channels[0], in_channels[0]),
            nn.LayerNorm(in_channels[0], elementwise_affine=False),
            nn.SiLU())
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()

        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
            self, x, edge_index, edge_attr=None, edgeweight=None,
            emb=None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        C = self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, C)
        key = self.lin_key(x[0]).view(-1, C)
        value = self.lin_value(x[0]).view(-1, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index, query=query, key=key, value=value,
            edge_attr=edge_attr, edgeweight=edgeweight, emb=emb, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            # x_r = self.lin_skip(x[1])
            x_r = x[1]
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i, key_j, value_j, edge_attr, edgeweight, emb, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out *= edgeweight

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class ClofNet(torch.nn.Module):
    r"""
        ClofNet

        Args:
            pos_require_grad (bool, optional): If set to :obj:`True`, will require to take derivative of model output
             with respect to the atomic positions. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`96`)
            y_mean (float, optional): Mean value of the labels of training data. (default: :obj:`0`)
            y_std (float, optional): Standard deviation of the labels of training data. (default: :obj:`1`)
    """

    def __init__(
            self, max_atomic_num: int, pos_require_grad=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, num_radial=96, y_mean=0, y_std=1, eps=1e-10, **kwargs):
        super(ClofNet, self).__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.eps = eps
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.pos_require_grad = pos_require_grad

        self.z_emb_ln = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.z_emb = Embedding(max_atomic_num, hidden_channels)

        self.radial_emb = RBFEmb(num_radial, self.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.neighbor_emb = NeighborEmb(hidden_channels)

        self.lin = nn.Sequential(
            nn.Linear(3, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, 1))

        self.edge_proj = nn.Sequential(
            nn.Linear(8, hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels // 4, hidden_channels)
        )

        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(
                TransformerConv(hidden_channels, hidden_channels, edge_dim=self.hidden_channels, **kwargs))
        # self.last_layer = nn.Linear(hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        # self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, z, pos, batch):
        # Note: the input pos needs to be centered at origin for translation equivariance
        if self.pos_require_grad:
            pos.requires_grad_()

        # embed z
        z_emb = self.z_emb_ln(self.z_emb(z))

        # construct edges based on the cutoff value
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        i, j = edge_index

        # embed pair-wise distance
        dist = torch.sqrt(torch.sum((pos[i] - pos[j]) ** 2, 1))
        # radial_emb shape: (num_edges, num_radial), radial_hidden shape: (num_edges, hidden_channels)
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        # build edge-wise frame
        edge_diff = pos[i] - pos[j]
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)
        edge_cross = torch.cross(pos[i], pos[j])
        edge_cross = edge_cross / ((torch.sqrt(torch.sum((edge_cross) ** 2, 1).unsqueeze(1))) + self.eps)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # shape: (num_edges, 3, 3)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)

        pos_i = pos[i]
        pos_j = pos[j]
        coff_i = torch.matmul(edge_frame, pos_i.unsqueeze(-1)).squeeze(-1)
        coff_j = torch.matmul(edge_frame, pos_j.unsqueeze(-1)).squeeze(-1)
        # Calculate angle information in local frames
        coff_mul = coff_i * coff_j  # [E, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) + 1e-5
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) + 1e-5
        pesudo_cos = coff_mul.sum(dim=-1, keepdim=True) / coff_i_norm / coff_j_norm
        pesudo_sin = torch.sqrt(1 - pesudo_cos ** 2)
        pesudo_angle = torch.cat([pesudo_sin, pesudo_cos], dim=-1)
        coff_feat = torch.cat([pesudo_angle, coff_i, coff_j], dim=-1)

        edgeweight = radial_hidden
        edgefeature = self.edge_proj(coff_feat)

        for i in range(self.num_layers):
            z_emb = self.message_layers[i](z_emb, edge_index, edge_attr=edgefeature, edgeweight=edgeweight)

        # s = self.last_layer(z_emb)
        s = scatter(z_emb, batch, dim=0)
        return s
