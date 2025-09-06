# QUantum Electronic Descriptors (QUED)


## Required libraries

#### To convert SMILES to 3D coordinates
```console
conda create -n qued python=3.9
conda activate qued
conda install -c conda-forge rdkit numpy scipy pandas matplotlib
```

#### To read QM7-X dataset
```console
pip install schnetpack==0.3
```

#### To read TOX and LIPO datasets
```console
conda install h5py
```

#### To generate BoB and SLATM
I tried the option indicated by documentation and krr-opt/experiments/QM7X/README.md but neither of them worked
```console
pip3 install qml
```


### Create virtual environment for KRR-OPT tool

#### Before the installation, create and activate your conda env with python 3.9:
```console
conda create --prefix /home/alhi416g/.conda/envs/krrOpt2 python=3.9
conda activate krrOpt2

conda install -c conda-forge 'joblib>=1.3.0' 'scipy>=1.11.0' 'numpy>1.23.0,<1.24.0' 'matplotlib>=3.7.0' 'scikit-learn>=1.5.0' 'typing-extensions>=4.7.0'
```

#### You will need the fixed fork of SHGO optimizer (v. 0.5.1):
```console
pip install -U git+https://github.com/arkochem/shgo.git@v0.5.1#egg=shgo
```

#### To install the krr-opt package: 
Open terminal from the source directory containing README.md, setup.py (files) 
and krr_opt (directory), and run the following (do not forget to activate your environment):
```console
python setup.py bdist_wheel
```
Then use generated .whl file (example below) to install the package with all dependencies via pip:
```console
pip install ./dist/krr_opt-<version>-py3-none-any.whl
```

### Create virtual environment for XGBoost training

#### Before the installation, create and activate your conda env with python 3.10:
```console
conda create --prefix /home/alhi416g/.conda/envs/xgbOpt python=3.10
conda activate xgbOpt

conda install numpy pandas matplotlib scikit-learn
conda install -c conda-forge optuna
conda install -c conda-forge shap
```

