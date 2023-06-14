import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from itertools import groupby
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.ensemble import EnsembleDataset


class EE_2D(EnsembleDataset):
    descriptors = ['de']
    excluded_ids = ['atrop-merg-enamide-phe-bn-h-B_R10333']

    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'EE_2D_processed.pt'

    @property
    def raw_file_names(self):
        return 'EE.zip'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    def process(self):
        data_list = []

        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        label_file = raw_file.replace('.zip', '.csv')
        labels = pd.read_csv(label_file)

        raw_file = raw_file.replace('.zip', '.sdf')
        mols = {}
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for mol in tqdm(suppl):
                id = mol.GetProp('id')
                substrate_id = mol.GetProp('substrate_id')
                ligand_id = mol.GetProp('ligand_id')
                config_id = mol.GetProp('config_id')
                name = '_'.join([substrate_id, ligand_id])

                if name in mols or name in self.excluded_ids:
                    continue

                frags = Chem.GetMolFrags(mol, asMols=True)
                if frags[0].GetNumAtoms() > frags[1].GetNumAtoms():
                    frags = frags[::-1]
                data_0 = mol_to_data_obj(frags[0])
                data_1 = mol_to_data_obj(frags[1])

                data_0.id = id
                data_0.smiles = Chem.MolToSmiles(frags[0])
                data_0.ligand_id = ligand_id
                data_0.substrate_id = substrate_id
                data_0.config_id = config_id
                data_0.is_ligand = 1
                data_0.name = name

                data_1.id = id
                data_1.smiles = Chem.MolToSmiles(frags[1])
                data_1.ligand_id = ligand_id
                data_1.substrate_id = substrate_id
                data_1.config_id = config_id
                data_1.is_ligand = 0
                data_1.name = name

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data_0 = self.pre_transform(data_0)
                    data_1 = self.pre_transform(data_1)

                mols[name] = (data_0, data_1)

        cursor = 0
        ys = []
        for name, mol_list in tqdm(mols.items()):
            y = labels[labels['MergeID'] == name][self.descriptors[0]].values[0]
            ys.append(y)
            for mol in mol_list:
                mol.molecule_idx = cursor
                data_list.append(mol)
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)

        data, slices = self.collate(data_list)
        torch.save((data, slices, ys), self.processed_paths[0])


class EE(EnsembleDataset):
    descriptors = ['de']
    excluded_ids = ['atrop-merg-enamide-phe-bn-h-B_R10333']

    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'EE_processed.pt' if self.max_num_conformers is None \
            else f'EE_{self.max_num_conformers}_processed.pt'

    @property
    def raw_file_names(self):
        return 'EE.zip'

    @property
    def num_molecules(self):
        return self.y.shape[0]

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        data_list = []

        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        label_file = raw_file.replace('.zip', '.csv')
        labels = pd.read_csv(label_file)

        raw_file = raw_file.replace('.zip', '.sdf')
        mols = defaultdict(list)
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for mol in tqdm(suppl):
                data = mol_to_data_obj(mol)

                data.energy = float(mol.GetProp('energy'))
                data.smiles = mol.GetProp('smiles')
                data.substrate_id = mol.GetProp('substrate_id')
                data.ligand_id = mol.GetProp('ligand_id')
                data.config_id = 1 if mol.GetProp('config_id') == '1' else 0
                data.id = mol.GetProp('id')
                data.name = data.substrate_id + '_' + data.ligand_id

                if data.name in self.excluded_ids:
                    continue
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mols[data.name].append(data)

        cursor = 0
        ys = []
        for name, mol_list in tqdm(mols.items()):
            y = labels[labels['MergeID'] == name][self.descriptors[0]].values[0]
            ys.append(y)
            mol_list = sorted(mol_list, key=lambda x: (x.config_id, x.energy))
            grouped_mol_list = [list(g) for k, g in groupby(mol_list, key=lambda x: x.config_id)]
            if self.max_num_conformers is not None:
                new_mol_list = []
                for mol_list_per_config in grouped_mol_list:
                    new_mol_list.append(mol_list_per_config[:self.max_num_conformers])
                mol_list = [mol for sublist in new_mol_list for mol in sublist]
            else:
                mol_list = [mol for sublist in grouped_mol_list for mol in sublist]

            for mol in mol_list:
                mol.molecule_idx = cursor
                data_list.append(mol)
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)

        data, slices = self.collate(data_list)
        torch.save((data, slices, ys), self.processed_paths[0])
