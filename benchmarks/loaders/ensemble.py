import copy
import torch
import numpy as np

from torch import Tensor
from typing import Sequence
from sklearn.utils import shuffle
from torch_geometric.data import InMemoryDataset


class EnsembleDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    def mean(self, target: str) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        target_id = self.descriptors.index(target)
        return float(y[:, target_id].mean())

    def std(self, target: str) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        target_id = self.descriptors.index(target)
        return float(y[:, target_id].std())

    def index_select(self, idx):
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError

        dataset = copy.copy(self)
        dataset._indices = indices

        # Update descriptors
        molecule_idx = torch.unique_consecutive(dataset.data.molecule_idx[indices])
        dataset.y = dataset.y[molecule_idx]

        # Update molecule indices
        previous_idx = dataset.data.molecule_idx[indices[0]].item()
        cursor = 0
        for i, idx in enumerate(dataset.data.molecule_idx[indices]):
            if idx.item() != previous_idx:
                cursor += 1
                previous_idx = idx.item()
            dataset.data.molecule_idx[indices[i]] = cursor
        return dataset

    def get_idx_split(self, train_ratio, valid_ratio, seed=None, max_num_molecules=None):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        if max_num_molecules is not None:
            molecule_ids = molecule_ids[:max_num_molecules]
            train_size = int(train_ratio * max_num_molecules)
            valid_size = int(valid_ratio * max_num_molecules)
        else:
            train_size = int(train_ratio * self.num_molecules)
            valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size:train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size:])

        molecule_idx = self.data.molecule_idx[self._indices]
        if self._indices is None:
            molecule_idx = molecule_idx.squeeze(0)
        train_conformer_idx = torch.where(torch.isin(molecule_idx, train_idx))[0]
        valid_conformer_idx = torch.where(torch.isin(molecule_idx, valid_idx))[0]
        test_conformer_idx = torch.where(torch.isin(molecule_idx, test_idx))[0]

        if max_num_molecules is None or max_num_molecules > self.num_molecules:
            assert len(molecule_idx) == \
                len(set(train_conformer_idx.numpy()).union(
                    set(valid_conformer_idx.numpy()), set(test_conformer_idx.numpy())))

        split_dict = {'train': train_conformer_idx, 'valid': valid_conformer_idx, 'test': test_conformer_idx}
        return split_dict

    def shuffle(self, return_perm=False):
        molecule_idx = self.data.molecule_idx[self._indices]
        all_molecules, molecule_counts = molecule_idx.unique(return_counts=True)
        cursor = 0
        molecule_conformer_mapping = []
        for molecule, count in zip(all_molecules, molecule_counts):
            molecule_conformer_mapping.append(list(range(cursor, cursor + count)))
            cursor += count
        perm = torch.randperm(self.num_molecules)
        index = []
        for i in perm:
            index += molecule_conformer_mapping[i]
        dataset = self.index_select(index)
        return (dataset, index) if return_perm is True else dataset

    def __repr__(self):
        return f'{self.__class__.__name__}: ' \
               f'{self.num_molecules} molecules, {self.num_conformers} conformers'
