import torch
import random

from rdkit import Chem
from itertools import chain
from collections import defaultdict
from torch_geometric.loader import DataLoader

from .utils import boltzmann_average
from data.drugs import Drugs


class EnsembleSampler:
    def __init__(self, dataset, batch_size, strategy='all', shuffle=True):
        assert strategy in ['all', 'random', 'first']
        self.molecule_idx = dataset.data.molecule_idx[dataset._indices]
        self.num_molecules = dataset.num_molecules
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle = shuffle

    def __iter__(self):
        all_molecules, molecule_counts = self.molecule_idx.unique(return_counts=True)
        cursor = 0
        molecule_conformer_mapping = []
        for molecule, count in zip(all_molecules, molecule_counts):
            molecule_conformer_mapping.append(list(range(cursor, cursor + count)))
            cursor += count
        assert self.num_molecules == len(molecule_conformer_mapping)

        if self.shuffle:
            index = torch.randperm(self.num_molecules)
        else:
            index = torch.arange(self.num_molecules)

        batch_index = []
        for i in index:
            if self.strategy == 'all':
                batch_index += molecule_conformer_mapping[i]
            elif self.strategy == 'random':
                batch_index.append(random.choice(molecule_conformer_mapping[i]))
            elif self.strategy == 'first':
                batch_index.append(molecule_conformer_mapping[i][0])

            if len(batch_index) >= self.batch_size:
                yield batch_index
                batch_index = []
        if len(batch_index) > 0:
            yield batch_index


class EnsembleMultiBatchSampler:
    def __init__(self, dataset, batch_size, variable_name, strategy='all', shuffle=True):
        assert strategy in ['all', 'random', 'first']
        self.molecule_idx = dataset.data.molecule_idx[dataset._indices]
        self.num_molecules = dataset.num_molecules
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle = shuffle
        self.variable_name = variable_name
        self.variables = dataset.data[variable_name][dataset._indices]

    def __iter__(self):
        all_molecules, molecule_counts = self.molecule_idx.unique(return_counts=True)
        cursor = 0
        molecule_conformer_mapping = []
        molecule_variable_mapping = []
        for molecule, count in zip(all_molecules, molecule_counts):
            molecule_conformer_mapping.append(list(range(cursor, cursor + count)))
            molecule_variable_mapping.append(self.variables[cursor:cursor + count])
            cursor += count
        assert self.num_molecules == len(molecule_conformer_mapping)

        if self.shuffle:
            index = torch.randperm(self.num_molecules)
        else:
            index = torch.arange(self.num_molecules)

        unique_variables = sorted(self.variables.unique().tolist())
        batch_indices = {var: [] for var in unique_variables}

        for i in index:
            conformer_indices = molecule_conformer_mapping[i]
            variables = molecule_variable_mapping[i]
            for var in unique_variables:
                conformer_ids_by_var = [idx for idx, id_var in zip(conformer_indices, variables) if id_var == var]

                if self.strategy == 'all':
                    batch_indices[var].extend(conformer_ids_by_var)
                elif self.strategy == 'random':
                    id_var = random.choice(conformer_ids_by_var)
                    batch_indices[var].append(id_var)
                elif self.strategy == 'first':
                    batch_indices[var].append(conformer_ids_by_var[0])

            if len(batch_indices[unique_variables[0]]) >= self.batch_size:
                yield [batch_indices[var] for var in unique_variables]
                batch_indices = {var: [] for var in unique_variables}
        if len(batch_indices[unique_variables[0]]) > 0:
            yield [batch_indices[var] for var in unique_variables]


if __name__ == '__main__':
    dataset = Drugs(root='../datasets/Drugs').shuffle()

    split = dataset.get_idx_split(train_ratio=0.1, valid_ratio=0.1)
    train_dataset = dataset[split['train']]
    valid_dataset = dataset[split['valid']]
    test_dataset = dataset[split['test']]
    test_loader = DataLoader(test_dataset, batch_sampler=EnsembleSampler(
        test_dataset, batch_size=32, shuffle=False))
    valid_loader = DataLoader(valid_dataset, batch_sampler=EnsembleSampler(
        valid_dataset, batch_size=32, shuffle=False))
    train_loader = DataLoader(train_dataset, batch_sampler=EnsembleSampler(
        train_dataset, batch_size=32, shuffle=True))

    dictionaries = {}
    with Chem.SDMolSupplier('../datasets/Drugs/raw/Drugs.sdf', removeHs=False) as suppl:
        import pdb; pdb.set_trace()
        for mol in suppl:
            id_ = mol.GetProp('ID')
            pos = mol.GetConformer().GetPositions()
            y = []
            for quantity in ['energy', 'ip', 'ea', 'chi', 'eta', 'omega']:
                y.append(float(mol.GetProp(quantity)))
            dictionaries[id_] = (pos, torch.Tensor(y))

    all_ids = set()
    all_names = set()
    conformer_labels = defaultdict(list)
    molecule_labels = {}

    for data in train_loader:
        unique_molecule_idx = torch.unique_consecutive(data.molecule_idx)
        for name, y in zip(data.name, data.y):
            conformer_labels[name].append(y)
        for id_ in unique_molecule_idx:
            name = data[data.molecule_idx == id_][0].name
            molecule_labels[name] = train_dataset.y[id_]

        for name in molecule_labels.keys():
            energy_list = [conformer[0].item() for conformer in conformer_labels[name]]
            for idx in range(6):
                quantity_list = [conformer[idx].item() for conformer in conformer_labels[name]]
                boltzmann_avg_quantity = boltzmann_average(quantity_list, energy_list)
                assert boltzmann_avg_quantity - molecule_labels[name][idx].item() < 1e-5

    for data in chain(train_loader, valid_loader, test_loader):
        all_ids |= set(data.id)
        all_names |= set(data.name)

        id_ = data.id[0]
        comp = data.pos[:2] == torch.from_numpy(dictionaries[id_][0])[:2]
        assert comp.all()

        comp = abs(data.y[0] - dictionaries[id_][1]) < 1e-5
        assert comp.all()

    assert len(all_ids) == len(dataset)
    assert len(all_names) == dataset.num_molecules
