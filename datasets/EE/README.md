# EE

## Statistics

| Dataset | Num. Reactions | Num. Conformers                 | Num. Heavy Atoms              | Num. Rotatable Bonds | Num. Targets | Atomic Species                                   |
| :------ | :------------- | :------------------------------ | :---------------------------- | :------------------- | :----------- | :----------------------------------------------- |
| EE      | 872            | Pro-R: 14,807, Pro-S: 13,999    | 59.32                         | 18.57                | 1            | H, C, N, O, F, P, Cl, Br, Rh                     |

## Download

[Google Drive](https://drive.google.com/file/d/1rIknilf5D-J50lPblsPtuFhEil9K1RO-/view?usp=sharing)

## Format

There are two files inside `EE.zip`:

* `EE.sdf`: The geometry of each catalyst-substrate transition state can be accessed there. Each transition state is indexed by `substrate_id` and `ligand_id` and labeled by a binary value `config_id`, where `0` denotes pro-R and `1` denotes pro-S configurations respectively.
* `EE.csv`: The enantiomeric excess values of each reaction are in the `de` column, indexed by `MergeID`.

## License

The EE dataset is publicly accessible. The copyright of the dataset is retained by the original authors. An archived copy is also available on Zenodo at https://doi.org/10.5281/zenodo.22212806.
