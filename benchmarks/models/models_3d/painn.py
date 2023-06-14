"""
https://github.com/atomistic-machine-learning/schnetpack/blob/dev/src/schnetpack/representation/painn.py
"""

import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.init import xavier_uniform_, zeros_
from torch_scatter import scatter
from torch_geometric.nn import radius_graph

from typing import Callable, Optional, Union


class Dense(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: Union[Callable, nn.Module] = None,
            weight_init: Callable = xavier_uniform_,
            bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


def scatter_add(
        x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    return _scatter_add(x, idx_i, dim_size, dim)


def _scatter_add(
        x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


def replicate_module(module_factory: Callable[[], nn.Module], n: int, share_params: bool):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for _ in range(n)])
    return module_list


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
            self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.
        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}
        Args:
            cutoff (float, optional): cutoff radius.
        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(
            self,
            q: torch.Tensor,
            mu: torch.Tensor,
            Wij: torch.Tensor,
            dir_ij: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor,
            n_atoms: int,
    ):
        """Compute interaction output.
        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j
        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.
        Args:
            q: scalar input values
            mu: vector input values
        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V ** 2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network
    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
            self,
            hidden_dim: int,
            num_interactions: int,
            num_rbf: int,
            cutoff: float,
            # n_out: int,
            readout: str,
            # n_out_hidden: int = None,
            # n_out_layers: int = 2,
            # radial_basis: nn.Module,
            # cutoff_fn: Optional[Callable] = None,
            activation: Optional[Callable] = F.silu,
            max_atomic_num: int = 100,
            shared_interactions: bool = False,
            shared_filters: bool = False,
            epsilon: float = 1e-8,
    ):
        """
        Args:
            hidden_dim: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            num_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_interactions = num_interactions
        self.activation = activation

        cutoff_fn = CosineCutoff(cutoff)
        radial_basis = GaussianRBF(n_rbf=num_rbf, cutoff=cutoff)
        self.cutoff = cutoff
        self.cutoff_fn = cutoff_fn
        self.radial_basis = radial_basis

        self.readout = readout
        self.embedding = nn.Embedding(max_atomic_num, hidden_dim, padding_idx=0)
        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = Dense(
                self.radial_basis.n_rbf, 3 * hidden_dim, activation=None
            )
        else:
            self.filter_net = Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * hidden_dim * 3,
                activation=None,
            )

        self.interactions = replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.hidden_dim, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.hidden_dim, activation=activation, epsilon=epsilon
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, x, pos, batch, return_latent=False):
        """
        Compute atomic representations/embeddings.
        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.
        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        atomic_numbers = x
        radius_edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        idx_i, idx_j = radius_edge_index[0], radius_edge_index[1]
        # r_ij = (positions[idx_i] - positions[idx_j])**2
        r_ij = pos[idx_i] - pos[idx_j]
        n_atoms = atomic_numbers.size()[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.hidden_dim, dim=-1)

        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        h = q
        h = scatter(h, batch, dim=0, reduce=self.readout)
        if return_latent:
            return h, q
        return h
