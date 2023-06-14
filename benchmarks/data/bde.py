import torch
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from collections import defaultdict
from torch_geometric.data import extract_zip

from loaders.utils import mol_to_data_obj
from loaders.ensemble import EnsembleDataset


class BDE(EnsembleDataset):
    descriptors = ['BindingEnergy']

    def __init__(self, root, max_num_conformers=None, transform=None, pre_transform=None, pre_filter=None):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self):
        return 'BDE_processed.pt' if self.max_num_conformers is None \
            else f'BDE_{self.max_num_conformers}_processed.pt'

    @property
    def raw_file_names(self):
        return 'BDE.zip'

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
        label_file = raw_file.replace('.zip', '.txt')
        labels = pd.read_csv(
            label_file, sep='  ', header=0,
            names=['Name', self.descriptors[0]], engine='python')

        filenames = ['substrates', 'ligands']
        mols = defaultdict(list)
        mols_count = [defaultdict(int), defaultdict(int)]
        for idx, filename in enumerate(filenames):
            raw_file = label_file.replace('BDE.txt', f'{filename}.sdf')
            with Chem.SDMolSupplier(raw_file, removeHs=True) as suppl:
                for mol in tqdm(suppl):
                    data = mol_to_data_obj(mol)

                    data.smiles = Chem.MolToSmiles(mol)
                    data.name = mol.GetProp('Name')
                    data.id = mol.GetProp('Index')
                    data.is_ligand = True if filename == 'ligands' else False

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if self.max_num_conformers is not None and mols_count[idx][data.name] > self.max_num_conformers:
                        continue
                    mols[data.name].append(data)
                    mols_count[idx][data.name] += 1

        cursor = 0
        ys = []
        for name, mol_list in tqdm(mols.items()):
            y = labels[labels['Name'] == name][self.descriptors[0]].values[0]
            ys.append(y)
            for mol in mol_list:
                mol.molecule_idx = cursor
                data_list.append(mol)
            cursor += 1
        ys = torch.Tensor(ys).unsqueeze(1)

        data, slices = self.collate(data_list)
        torch.save((data, slices, ys), self.processed_paths[0])
