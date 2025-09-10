# QUantum Electronic Descriptors (QUED)

## Required libraries

#### Create and activate your `conda env` with `python 3.9`:
```bash
conda create --prefix /home/alhi416g/.conda/envs/qued python=3.9
conda activate qued

conda install -c conda-forge 'joblib>=1.3.0' 'scipy>=1.11.0' 'numpy>1.23.0,<1.24.0' 'matplotlib>=3.7.0' 'scikit-learn>=1.5.0' 'typing-extensions>=4.7.0'
```

#### Install `rdkit` to convert SMILES to 3D coordinates
```bash
conda install -c conda-forge rdkit
conda install pandas
```

#### Install `crest` for conformational search
```bash
conda install conda-forge::crest
```

#### Install `dftb+` and `ase` to compute QM properties
```bash
# Note: DFTB+ downgrades xTB (6.7.1. -> 6.6.1.)
conda install -n qued mamba
mamba install 'dftbplus=*=mpi_openmpi_*'
# additional components like the dptools and the Python API
mamba install dftbplus-tools dftbplus-python
conda install conda-forge::ase
# the dftb.py file has to be replaced in the calculators module of ase
# to include calculated reference values for Hubbard Derivatives
cp dftb.py /home/alhi416g/.conda/envs/qued/lib/python3.9/site-packages/ase/calculators/dftb.py
```

#### Install `qml` to generate BoB and SLATM
```bash
pip3 install qml
```

### KRR-OPT tool

#### You will need the fixed fork of `SHGO optimizer (v. 0.5.1)`:
```bash
pip install -U git+https://github.com/arkochem/shgo.git@v0.5.1#egg=shgo
```

#### To install the `krr-opt` package: 
Open terminal from the source directory containing README.md, setup.py (files) 
and krr_opt (directory), and run the following (do not forget to activate your environment):
```bash
python setup.py bdist_wheel
```
Then use generated .whl file (example below) to install the package with all dependencies via pip:
```bash
pip install ./dist/krr_opt-<version>-py3-none-any.whl
```

### XGBoost training
```bash
conda install -c conda-forge py-xgboost
conda install h5py
conda install -c conda-forge optuna
conda install -c conda-forge shap
```

## Other libraries

#### To read QM7-X dataset
```bash
pip install schnetpack==0.3
```

#### To read TOX and LIPO datasets
```bash
conda install h5py
```

## Run scripts

### Convert SMILES to 3D coordinates
Example with ld50-tdcommons.csv database
```bash
conda activate qued
python3 smile2geom.py -i ld50-tdcommons.csv -x 'Drug' -y 'Y'
```

### Perform conformational search
Example with 1301.xyz (1-Butanethiol -> CCCCS)
```bash
# path where initial xyz file is stored
molid=1301
mol=/data/horse/ws/alhi416g-qmbio2/QUED/tests/geometries/${molid}.xyz

# temporal directory to run CREST
mkdir -p tmp
cd tmp
# conformational search for small molecules
crest $mol -gfn2 -gbsa h2o -mrest 10 -rthr 0.1 -ewin 12.0
# conformational search for medium-size molecules
crest $mol -gfn2 -gbsa h2o -opt normal -quick -mrest 5 -rthr 0.1 -ewin 12.0
# conformational search for large molecules
crest $mol -gfn2 -gbsa h2o -opt lax -norotmd -mquick -mrest 5 -rthr 0.1 -ewin 12.0
# exit temporal directory
cd ..

mkdir -p crest_conformers
mv tmp/crest_conformers.xyz conformers/${molid}.xyz
rm -r tmp
```

### Calculate QM properties
Elements covered by the 3ob parameters set: Br-C-Ca-Cl-F-H-I-K-Mg-N-Na-O-P-S-Zn. Download parameters from [dft.org](https://dftb.org/parameters/download.html#)
Takes only the top10 conformers with the lowest xTB energy after the generation
```bash
# change this as necessary
module load release/23.10 GCC/12.2.0 OpenMPI/4.1.4 Anaconda3/2023.07-2
source activate /home/alhi416g/.conda/envs/qued
export DFTB_COMMAND='mpiexec -n 1 /home/alhi416g/.conda/envs/qued/bin/dftb+'
export DFTB_PREFIX='/data/horse/ws/alhi416g-qmbio2/QUED/3ob-3-1/'
export OMP_NUM_THREADS=1

mol=/data/horse/ws/alhi416g-qmbio2/QUED/tests/conformers/${molid}.xyz

python3 qmcalc.py -i $mol -n 10 -o qmprops

# remove files from dftb+
rm band.out charges.bin current_dftb.out detailed.out dftb_in.hsd dftb_pin.hsd geo_end.gen
```


