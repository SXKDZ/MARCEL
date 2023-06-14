import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def construct_fingerprint(smiles):
    mol = [Chem.MolFromSmiles(s) for s in smiles]

    # Morgan fingerprint
    mfp = np.asarray(
        [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=3 * 1024) for x in mol], dtype=np.int8)

    # RDKit topological fingerprint
    rdkbi = {}
    rdfp = np.asarray([Chem.RDKFingerprint(x, maxPath=5, bitInfo=rdkbi) for x in mol], dtype=np.int8)

    # MACCS keys
    maccs = np.asarray([MACCSkeys.GenMACCSKeys(x) for x in mol], dtype=np.int8)

    fingerprint = np.concatenate([mfp, rdfp, maccs], axis=1)

    # dropout redundant features
    correlation = np.corrcoef(np.transpose(fingerprint))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    correlation[mask] = 0
    drop = np.where(abs(correlation) > 0.9)[1]
    return np.delete(fingerprint, drop, axis=1)


def construct_smiles(dataset):
    molecule_idx = dataset.data.molecule_idx[dataset._indices]
    molecules, counts = molecule_idx.unique(return_counts=True)
    cursor = 0
    mapping = []
    for count in counts:
        mapping.append(list(range(cursor, cursor + count)))
        cursor += count
    conformer_index = [mapping[i][0] for i in range(len(molecules))]
    dataset = dataset[conformer_index]
    return dataset.smiles


def concatenate_smiles(dataset, variable_name):
    molecule_idx = dataset.data.molecule_idx[dataset._indices]
    variables = dataset.data[variable_name][dataset._indices]
    unique_variables = sorted(variables.unique().tolist())

    molecules, counts = molecule_idx.unique(return_counts=True)
    cursor = 0
    conformer_mapping = []
    variable_mapping = []
    for count in counts:
        conformer_mapping.append(list(range(cursor, cursor + count)))
        variable_mapping.append(variables[cursor:cursor + count])
        cursor += count

    all_smiles = []
    for i in range(len(molecules)):
        conformer_index = conformer_mapping[i]
        variable_by_conformer = variable_mapping[i]

        smiles = []
        for var in unique_variables:
            for idx, id_var in zip(conformer_index, variable_by_conformer):
                if id_var == var:
                    smiles.append(dataset.data.smiles[dataset._indices[idx]])
                    break
        all_smiles.append('.'.join(smiles))
    return all_smiles
