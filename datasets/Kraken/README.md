# Kraken

## Statistics

| Dataset | Num. Molecules | Num. Conformers | Num. Heavy Atoms | Num. Rotatable Bonds | Num. Targets | Atomic Species                                    |
| :------ | :------------- | :-------------- | :--------------- | :------------------- | :----------- | :------------------------------------------------ |
| Kraken  | 1,552          | 21,287          | 23.70            | 9.05                 | 4            | H, B, C, N, O, F, Si, P, S, Cl, Fe, Se, Br, Sn, I |

## Download

[Google Drive](https://drive.google.com/file/d/1QrV651Re7s6UF7Lg4KC9PM5QQUMPU7wd/view?usp=sharing)

## Format

Unzip `Kraken.zip` and load `Kraken.pickle` using Pickle:

```python
kraken = pickle.load(open('Kraken.pickle', 'rb'))
```

`kraken` is a `dict` indexed by `ligand_ids`:

```python
ligand_ids = list(kraken.keys())
```

For each molecule, the SMILES string, Boltzmann-averaged descriptors, and conformer geometry are given:

```python
smiles, boltz_avg_properties, conformer_dict = kraken[ligand_ids[0]]

# example boltzmann-averaged property
sterimol_B1_boltz_avg = boltz_avg_properties['sterimol_B5']
```

For each conformer in `conformer_dict`, you can access the `sdf` file for the conformer geometry, Boltzmann weight, and conformer-level descriptors:

```python
conformer_ids = list(conformer_dict.keys())

# accessing sdf (mol) file, boltzmann weight, and properties of the first conformer
mol_sdf, boltz_weight, conformer_properties = conformer_dict[conformer_ids[0]]

# converting sdf into rdkit mol object
mol = rdkit.Chem.MolFromMolBlock(mol_sdf, removeHs=False)
mol_noHs = rdkit.Chem.RemoveHs(mol)
```

## License

The Kraken dataset is publicly accessible at https://kraken.cs.toronto.edu. The copyright of the dataset is retained by the original authors. Access to the Kraken dataset is granted through the permission of the authors.

