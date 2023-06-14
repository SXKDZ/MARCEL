# BDE

## Statistics

| Dataset | Num. Reactions | Num. Conformers                 | Num. Heavy Atoms              | Num. Rotatable Bonds | Num. Targets | Atomic Species                                   |
| :------ | :------------- | :------------------------------ | :---------------------------- | :------------------- | :----------- | :----------------------------------------------- |
| BDE     | 5,915          | Ligand: 73,834, Complex: 40,264 | Ligand: 29.62, Complex: 32.38 | 6.99                 | 1            | H, C, N, O, F, P, Cl, Ni, Cu, Br, Pd, Ag, Pt, Au |

## Download

[Google Drive](https://drive.google.com/file/d/1CWgFDQcwPWLV3a555V8D8soLA1AXoRD6/view?usp=sharing)

## Format

There are three files after unzipping `BDE.zip`:

* `BDE.txt` records the binding energy of each reaction, indexed by `Name`.
* `ligands.sdf` stores the geometry of the unbound ligands, each indexed by `Name` and `Index`.
* `substrates.sdf` stores the geometry of the bound complexes, each indexed by `Name` and `Index`.

## License

The raw binding energy descriptors can be accessed at https://archive.materialscloud.org/record/2018.0014/v1 under the Creative Commons Attribution 4.0 International license. The conformers included in MARCEL are under the Apache license.