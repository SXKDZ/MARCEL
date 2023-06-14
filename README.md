![](banner.svg)

**MARCEL** is a PyTorch-based benchmark library that evaluates the potential of machine learning on conformer ensembles across a diverse set of molecules, datasets, and models.

# Why Learning over Conformer Ensembles?

It is critical to recognize that in reality molecules are not rigid, static objects; rather, thermodynamically-permissible rotations of chemical bonds, small vibrational motions, and dynamic intermolecular interactions cause molecules to continuously convert between different conformations. As a consequence, many experimentally observable chemical properties depend on the full distribution of thermodynamically-accessible conformers. Also, it is often challenging to determine *a priori* the conformers that predominantly contribute to molecular properties without doing prohibitively expensive simulations. Therefore, it is important to investigate the *collective* power of many different conformer structures lying on the local minima of the potential energy surface, also known as the *conformer ensemble*, for improving molecular representation learning models.

<p align="center">
<img src="https://media.drugdesign.org/course/molecular-geometry/conformers.gif" width="35%" alt="Copyright © 2022 drugdesign.org" class="center" alt="logo"/>
</p>

# Datasets

MARCEL include four datasets that cover a diverse range of chemical space, which focuses on four chemically-relevant tasks for both molecules and reactions, with an emphasis on Boltzmann-averaged properties of conformer ensembles computed at the Density-Functional Theory (DFT) level.

## Drugs-75K

Drugs-75K is a subset of the [GEOM-Drugs](https://github.com/learningmatter-mit/geom) dataset, which includes 75,099 molecules with at least 5 rotatable bonds. For each molecule, Auto3D is used to generate and optimize the conformer ensembles and AIMNet-NSE is used to calculate three important DFT-based reactivity descriptors: ionization potential, electron affinity, and electronegativity.

Links: [Download](https://drive.google.com/file/d/1PHgbrxdDyyjnxSUA71zXWG8WNwocDLPH/view?usp=sharing), [Instructions](datasets/Drugs)

## Kraken

Kraken is a dataset of 1,552 monodentate organophosphorus (III) ligands along with their DFT-computed conformer ensembles. We consider four 3D catalytic ligand descriptors exhibiting significant variance among conformers: Sterimol B~5~, Sterimol L, buried Sterimol B~5~, and buried Sterimol L. These descriptors quantify the steric size of a substituent in Å, and are commonly employed for Quantitative Structure-Activity Relationship (QSAR) modeling. The buried Sterimol variants describe the steric effects within the first coordination sphere of a metal.

Links: [Download](https://drive.google.com/file/d/1QrV651Re7s6UF7Lg4KC9PM5QQUMPU7wd/view?usp=sharing), [Instructions](datasets/Kraken)

## EE

EE is a dataset of 872 catalyst-substrate pairs involving 253 Rhodium (Rh)-bound atropisomeric catalysts derived from chiral bisphosphine, with 10 enamides as substrates. The dataset includes conformations of catalyst-substrate transition state complexes in two separate pro-S and pro-R configurations. The task is to predict the Enantiomeric Excess (EE) of the chemical reaction involving the substrate, defined as the absolute ratio between the concentration of each enantiomer in the product distribution. This dataset is generated with Q2MM, which automatically generates Transition State Force Fields (TSFFs) in order to simulate the conformer ensembles of each prochiral transition state complex. EE can then be computed from the conformer ensembles by Boltzmann-averaging the activation energies for the competing transition states. Unlike properties in Drugs-75K and Kraken, EE depends on the conformer ensembles of *each* pro-R and pro-S complex.

Links: Dataset access not publicly available, [Instructions](datasets/EE)

## BDE

BDE is a dataset containing 5,915 organometallic catalysts ML₁L₂ consisting of a metal center (M = Pd, Pt, Au, Ag, Cu, Ni) coordinated to two flexible organic ligands (L₁ and L₂), each selected from a 91-membered ligand library. The data includes conformations of each unbound catalyst, as well as conformations of the catalyst when bound to ethylene and bromide after oxidative addition with vinyl bromide. Each catalyst has an electronic binding energy, computed as the difference in the minimum energies of the bound-catalyst complex and unbound catalyst, following the DFT-optimization of their respective conformer ensembles. Although the binding energies are computed via DFT, the conformers provided for modeling are generated with Open Babel. This realistically represents the setting in which precise conformer ensembles are unknown at inference.

Links: [Download](https://drive.google.com/file/d/1CWgFDQcwPWLV3a555V8D8soLA1AXoRD6/view?usp=sharing), [Instructions](datasets/BDE)

# Benchmarks

## Prerequisites

The following packages are required for running the benchmarks.

* `pytorch >= 1.13.1`
* `pyg >= 2.0`
* `rdkit`
* `nni`
* `ogb`

## Dataset Loaders

MARCEL has implemented PyG data loaders for each dataset. Download the dataset and place each zipped file under its corresponding directory, i.e. `datasets/<NAME>/raw`.

| Dataset   | Dataloader class                                           |
| --------- | ---------------------------------------------------------- |
| Drugs-75K | `data.drugs.Drugs`                                         |
| Kraken    | `data.kraken.Kraken`                                       |
| EE        | `data.ee.EE_2D` for 2D models, `data.ee.EE` for the others |
| BDE       | `data.bde.BDE`                                             |

## Batch Samplers

For Drugs-75K and Kraken, use `EnsembleSampler` to sample mini-batches of molecules from `loaders.samplers`. You can specify the sampling strategy to `random` that randomly samples one conformer, `first` that always loads the first conformer in each ensemble, or `all` that loads all conformers.

Since EE and BDE involve interactions between two molecules, we implement another sampler `EnsembleMultiBatchSampler` from `loaders.samplers`. In this case, each conformer of the system will be loaded as a tuple `[data_0, data_1]`, which corresponds to one of the two molecules in the system.

## Instructions on Reproducing Results

The hyperparameters are stored in `benchmarks/params` folder.



## License

The MARCEL benchmarks are licensed under Apache 2.0 License.

