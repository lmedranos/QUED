[![paper-link](https://img.shields.io/badge/preprint-ChemRxiv-red.svg?style=flat-squar)](https://chemrxiv.org/engage/chemrxiv/article-details/68c61dd73e708a7649eb1250)

# QUantum Electronic Descriptors (QUED)

## About
QUED is a machine learning framework for molecular property prediction that integrates both structural and electronic information, enabling efficient modeling of the properties of drug-like molecules. It derives a **quantum-mechanical descriptor** from molecular and atomic properties using the semi-empirical DFTB method and combines it with inexpensive two-body and three-body **geometric descriptors** (BoB and SLATM) to create comprehensive molecular representations for training regression models (e.g., Kernel Ridge Regression, XGBoost). We validated QUED on the prediction of physicochemical properties (QM7-X dataset) and the ADMET endpoints: toxicity (TDCommons-LD50 dataset) and lipophilicity (MoleculeNet-Lipophilicity benchmark).

## QM datasets
The structural and property data of drug-like molecules in TDCommons-LD50 and MoleculeNet-Lipophilicity datasets can be downloaded from the ZENODO repository.

[![ZENODO](https://zenodo.org/badge/DOI/10.5281/zenodo.10208010.svg)](https://zenodo.org/records/17106019)

## Installation
QUED requires a `conda` environment with `python 3.9`.
In addition to well-known libraries, `qml` is installed to generate BoB and SLATM descriptors. Other useful packages for posterior dataset generation should also be installed.
```bash
conda create -n qued python=3.9
conda activate qued
pip3 install qml
conda install -c conda-forge 'joblib>=1.3.0' 'scipy>=1.11.0' 'numpy>1.23.0,<1.24.0' 'matplotlib>=3.7.0' 'scikit-learn>=1.5.0'
conda install pandas
conda install h5py
```

Install `rdkit` for conversion of SMILES to 3D coordinates
```bash
conda install -c conda-forge rdkit
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

It is necessary to replace the `dftb.py` file of the `ase` package with the one provided in this repository. This includes calculated reference values for Hubbard Derivatives.
```bash
conda install conda-forge::ase
cp dftb.py /path/to/.conda/envs/qued/lib/python3.9/site-packages/ase/calculators/dftb.py
```

The following packages are needed for training XGBoost models
```bash
conda install -c conda-forge py-xgboost optuna shap
```
Finally, install the `KRR-OPT` tool from the [krr-opt repository](https://github.com/arkochem/krr-opt.git).

## Command Line Interface
The `qued` directory include scripts that can be used as standalone executables. 

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
crest mol.xyz -gfn2 -gbsa h2o -mrest 5 -rthr 0.1 -ewin 12.0 -quick
# conformational search for large molecules
crest mol.xyz -gfn2 -gbsa h2o -mrest 5 -rthr 0.1 -ewin 12.0 -mquick -norotmd
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

Takes up to 10 conformers with the lowest xTB energy after the generation and creates `Geom-n[i]-c[j].npz` files, which store the calculated QM properties. 
```bash
python3 electronic.py -i crest_conformers.xyz -n 10 -o qmprops
```

### Dataset generation
The script `smile2database.py` allows the user to input a SMILE or a csv file with SMILEs to directly generate a HDF5 file with the DFTB-calculated properties of up to `n` conformers obtained by CREST conformational search (when `n=0`, it extracts all conformers), and optionally include a target property present in the csv file:
```bash
python3 smile2database.py -i dataset.csv -x 'SMILE' -y <optional target> -o <output directory> -n 0
```
By default, this program performs conformational search considering this configuration: `-gfn2 -gbsa h2o -mrest 5 -rthr 0.1 -ewin 12.0 -mquick`. In case the user decide to use a different setting, the line 84 of this script should be modified with the desired arguments.


### Validate trained ML regression models
The hyperparameters of trained ML models are included in pickle files in the `models` directory. The user can choose between XGBoost and KRR models trained for toxicity and lipophilicity prediction (we include only the best models per dataset and per regression model). The script `dataset2pred` allows the user to use these trained models to 1) Validate the results resported in the paper. 2) Perform inference in a new dataset of molecules. The flag `-v` serves for the objective 1), in this case, the user must include the employed training dataset (in HDF5 format), which can be found in the models directory (decompress the zip file before running this script). 

In the case of XGBoost, the trained pipeline is already included in the pickle file.
```bash
# validation
python3 dataset2pred.py -v -i /path/to/training_dataset.h5 -m /path/to/model.pkl
# inference
python3 dataset2pred.py -i /path/to/new_dataset.h5 -m /path/to/model.pkl
```

It is not convenient to record the KRR estimator in a pickle file (it would result in a GB-sized file), so instead we saved all necessary parameters of the KRR model in the pickle file and re-train (or re-fit) the KRR model with the employed training dataset (in HDF5 format). Such training datasets can be found in the `models` directory (decompress the zip file before running this script). 
```bash
# validation
python3 dataset2pred.py -v -i /path/to/training_dataset.h5 -m /path/to/model.pkl -t /path/to/training_dataset.h5
# inference
python3 dataset2pred.py -i /path/to/new_dataset.h5 -m /path/to/model.pkl -t /path/to/training_dataset.h5
```
