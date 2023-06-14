# EE

## Statistics

| Dataset | Num. Reactions | Num. Conformers                 | Num. Heavy Atoms              | Num. Rotatable Bonds | Num. Targets | Atomic Species                                   |
| :------ | :------------- | :------------------------------ | :---------------------------- | :------------------- | :----------- | :----------------------------------------------- |
| EE      | 872            | Pro-R: 14,807, Pro-S: 13,999    | 59.32                         | 18.57                | 1            | H, C, N, O, F, P, Cl, Br, Rh                     |

## Download

Dataset access not publicly available

## Format

There are two files inside `EE.zip`:

* `EE.sdf`: The geometry of each catalyst-substrate transition state can be accessed there. Each transition state is indexed by `substrate_id` and `ligand_id` and labeled by a binary value `config_id`, where `0` denotes pro-R and `1` denotes pro-S configurations respectively.
* `EE.csv`: The enantiomeric excess values of each reaction are in the `de` column, indexed by `MergeID`.

## License

As of now, the EE dataset is proprietary, given that the publication addressing the conformer ensembles is still under preparation. Therefore, access to the EE dataset is restricted to review purposes only. We anticipate making the EE dataset publicly accessible following the acceptance of the corresponding paper.
