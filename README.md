[![paper-link](https://img.shields.io/badge/preprint-ChemRxiv-red.svg?style=flat-squar)](https://chemrxiv.org/engage/chemrxiv/article-details/68c61dd73e708a7649eb1250)

# QUantum Electronic Descriptors (QUED)

## About
QUED is a machine learning framework for molecular property prediction that integrates both structural and electronic information, enabling efficient modeling of the properties of drug-like molecules. It derives a **quantum-mechanical descriptor** from molecular and atomic properties using the semi-empirical DFTB method and combines it with inexpensive two-body and three-body **geometric descriptors** (BoB and SLATM) to create comprehensive molecular representations for training regression models (e.g., Kernel Ridge Regression, XGBoost). We validated QUED on the prediction of physicochemical properties (QM7-X dataset) and the ADMET endpoints: toxicity (TDCommons-LD50 dataset) and lipophilicity (MoleculeNet-Lipophilicity benchmark).

## QM datasets
The structural and property data of drug-like molecules in TDCommons-LD50 and MoleculeNet-Lipophilicity datasets can be downloaded from the ZENODO repository.

[![ZENODO](https://zenodo.org/badge/DOI/10.5281/zenodo.10208010.svg)](https://zenodo.org/records/17106019)

## Installation
QUED requires a `conda` environment with `python 3.9`. 
```bash
conda create -n qued python=3.9
conda activate qued
conda install -c conda-forge 'joblib>=1.3.0' 'scipy>=1.11.0' 'numpy>1.23.0,<1.24.0' 'matplotlib>=3.7.0' 'scikit-learn>=1.5.0'
```

Install `rdkit` for conversion of SMILES to 3D coordinates
```bash
conda install -c conda-forge rdkit
conda install pandas
```

Install `crest` for performing conformational search
```bash
conda install conda-forge::crest
```

Install `dftb+` and `ase` to compute QM properties. 
```bash
conda install -n qued mamba
mamba install 'dftbplus=*=mpi_openmpi_*'
# additional components like the dptools and the Python API
mamba install dftbplus-tools dftbplus-python
```
> **Note**: Although DFTB+ downgrades xTB (6.7.1. -> 6.6.1.), CREST runs normally.

It is necessary to replace the `dftb.py` file of the `ase` package with the one provided in this repository. This includes calculated reference values for Hubbard Derivatives.
```bash
conda install conda-forge::ase
cp dftb.py /path/to/.conda/envs/qued/lib/python3.9/site-packages/ase/calculators/dftb.py
```

Install `qml` to generate BoB and SLATM descriptors.
```bash
pip3 install qml
```

The following packages are needed for training XGBoost models
```bash
conda install -c conda-forge py-xgboost optuna shap
conda install h5py
pip install pyyaml
```
Finally, install the `KRR-OPT` tool from the [krr-opt repository](https://github.com/arkochem/krr-opt.git).

## Command Line Interface

### Convert SMILES to 3D coordinates
From a CSV file with SMILES (and optionally target property values), it creates xyz files for each molecule in the dataset with initial 3D atomic coordinates.
```bash
python3 smile2geom.py -i dataset.csv -x 'smiles' -y 'target' -o 'conformers'
```

### Perform conformational search
From the initial 3D atomic coordinates of a given molecule (stored in an xyz file), conformers are generated via CREST. The output file `crest_conformers.xyz` contains the atomic coordinates of all the generated conformers.
```bash
# conformational search for small molecules
crest mol.xyz -gfn2 -gbsa h2o -mrest 10 -rthr 0.1 -ewin 12.0
# conformational search for medium-size molecules
crest mol.xyz -gfn2 -gbsa h2o -opt normal -quick -mrest 5 -rthr 0.1 -ewin 12.0
# conformational search for large molecules
crest mol.xyz -gfn2 -gbsa h2o -opt lax -norotmd -mquick -mrest 5 -rthr 0.1 -ewin 12.0
```

### Calculate QM properties
Elements covered by the 3ob parameters set: Br-C-Ca-Cl-F-H-I-K-Mg-N-Na-O-P-S-Zn. The parameters can be downloaded from [dft.org](https://dftb.org/parameters/download.html#). Add the following environment variables to your shell (e.g., in .bashrc, .zshrc, or your job script), replacing the placeholder paths with your own installation paths:
```bash
# Path to your DFTB+ executable (MPI-enabled if available)
export DFTB_COMMAND='mpiexec -n 1 /path/to/dftb+/bin/dftb+'
# Path to the DFTB+ parameter set (e.g., 3ob-3-1 directory)
export DFTB_PREFIX='/path/to/slater-koster-files/3ob-3-1/'
# Set number of OpenMP threads (must be 1 if running under MPI)
export OMP_NUM_THREADS=1
```

Takes only the 10 conformers with the lowest xTB energy after the generation and creates `Geom-n[i]-c[j].npz` files, which store the calculated QM properties. 
```bash
python3 qmcalc.py -i crest_conformers.xyz -n 10 -o qmprops
```

### Validate trained ML regression models
The hyperparameters and parameters of trained ML models are included in npz files in the `models` directory. The user can choose between XGBoost and KRR models trained for toxicity and lipophilicity prediction (we include only the best models per dataset and per regression model). It requires the employed training dataset (in HDF5 format), which can be found in the `models` directory (decompress the zip file before running this script). For example, to validate the XGBoost model with BOB+QM descriptor for toxicity prediction:

```bash
python3 qmcalc.py -i crest_conformers.xyz -n 10 -r 'bob' -q -t ld50-stable-descriptors.npz -a 'xgboost' -p models/tox-xgb-bobqm-5k.npz
```
