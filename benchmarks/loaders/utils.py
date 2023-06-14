import torch
import numpy as np
import networkx as nx

from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data

from loaders.features import atom_to_feature_vector, bond_to_feature_vector


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def boltzmann_average(quantities, energies, k=8.617333262145e-5, temperature=298):
    # k: Boltzmann constant in eV/K
    assert len(quantities) == len(energies)
    if len(energies) == 1:
        return quantities[0]

    return np.sum(softmax(-np.asarray(energies) / k / temperature) * np.asarray(quantities))


def reorder_molecule_idx(molecule_idx):
    previous_idx = molecule_idx[0].item()
    cursor = 0
    new_molecule_idx = torch.zeros_like(molecule_idx).long()

    for i, idx in enumerate(molecule_idx[1:], 1):
        if idx.item() != previous_idx:
            cursor += 1
            previous_idx = idx.item()
        new_molecule_idx[i] = cursor

    return new_molecule_idx


def canonicalize_3d_mol(mol_smiles, mol_3d):
    def get_node_features(atomic_numbers):
        node_features = np.zeros((len(atomic_numbers), 100))
        for node_index, node in enumerate(atomic_numbers):
            features = np.zeros(100)  # one-hot atomic numbers
            features[node] = 1.
            node_features[node_index, :] = features
        return np.array(node_features, dtype=np.float32)

    def get_reindexing_map(g1, g2, use_node_features=True):
        if use_node_features:
            nm = nx.algorithms.isomorphism.generic_node_match(['Z'], [None], [np.allclose])
            gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2, node_match=nm)
        else:
            gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()  # THIS NEEDS TO BE CALLED FOR gm.mapping to be initiated
        idx_map = gm.mapping

        return idx_map

    def mol_to_nx(mol):
        m_atoms = mol.GetAtoms()
        m_atom_numbers = [a.GetAtomicNum() for a in m_atoms]
        adj = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol), dtype=int)

        node_feats = get_node_features(m_atom_numbers)
        node_feats_dict = {j: node_feats[j] for j in range(node_feats.shape[0])}

        g = nx.Graph(adj)
        nx.set_node_attributes(g, node_feats_dict, 'Z')

        return g

    try:
        mol_smiles = Chem.AddHs(mol_smiles)
        Chem.AllChem.EmbedMolecule(mol_smiles)
        mol_smiles.GetConformer()
        # check to make sure a conformer was actually generated
        # sometime conformer generation fails
    except:
        print('Failed to embed conformer')
        return None

    g_smiles = mol_to_nx(mol_smiles)
    g_3d = mol_to_nx(mol_3d)

    idx_map = get_reindexing_map(g_smiles, g_3d, use_node_features=True)

    new_3d_mol = deepcopy(mol_smiles)
    xyz_coordinates = mol_3d.GetConformer().GetPositions()
    for k in range(new_3d_mol.GetNumAtoms()):
        x, y, z = xyz_coordinates[idx_map[k]]
        new_3d_mol.GetConformer().SetAtomPosition(k, Point3D(x, y, z))

    return new_3d_mol


def mol_to_data_obj(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_feature = bond_to_feature_vector(bond)
            edge_attr.append(edge_feature)
            edge_attr.append(edge_feature)
        edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.asarray(edge_attr), dtype=torch.long)

    # coordinates
    pos = mol.GetConformer().GetPositions()
    pos = torch.from_numpy(pos).float()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data
