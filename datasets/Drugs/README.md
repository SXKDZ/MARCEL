# Drugs-75K

## Statistics

| Dataset | Num. Molecules | Num. Conformers | Num. Heavy Atoms | Num. Rotatable Bonds | Num. Targets | Atomic Species              |
| :------ | :------------- | :-------------- | :--------------- | :------------------- | :----------- | :-------------------------- |
| Drugs   | 75,099         | 558,002         | 30.56            | 7.53                 | 3            | H, C, N, O, F, Si, P, S, Cl |

## Download

[Google Drive](https://drive.google.com/file/d/1PHgbrxdDyyjnxSUA71zXWG8WNwocDLPH/view?usp=sharing)

## Format

There are two files in `Drugs.zip`:

* `Drugs.sdf`: The conformer geometry of each molecule. Each molecule and its associated conformers is indexed by a unique `name`, and each conformer is indexed by a unique `ID`. Conformer-level properties including `energy`, `ip`, `ea`, and `chi` are also provided.
* `Drugs.csv`: Boltzmann-averaged quantities of each molecule, indexed by the name of each molecule.

## License

The original GEOM-Drugs dataset is publicly available at https://github.com/learningmatter-mit/geom but no license is specified. Our generated conformer ensembles and descriptors are licensed under the Apache License.