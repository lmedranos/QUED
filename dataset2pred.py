"""
Make predictions with trained ML models
"""

from argparse import ArgumentParser

import os
import time
import pickle
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from krr_opt import *

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import shap

from qued.electronic import *
from qued.geometric import *

# read HDF5 file and extract properties
def prepare_data(dataset_path, representation, add_qm, target_property, configuration):
    # read file
    dataset = h5py.File(dataset_path, 'r')
    print(f"Reading {dataset_path}")

    # collecting properties
    xyz, Z, target = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = \
        [], [], [], [], [], [], [], [], [], [], []
    
    for molid in dataset.keys():
        conformers = dataset.get(molid)
        for confid in conformers.keys():
            properties = conformers.get(confid)
            # coordinates and nuclear charges
            xyz.append(np.array(properties['xyz'][()]))
            Z.append(np.array(properties['Z'][()]))
            # global
            p1.append(float(properties['FermiEne'][()]))
            p2.append(float(properties['BandEne'][()]))
            p3.append(float(properties['NumElec'][()]))
            p4.append(float(properties['h0Ene'][()]))
            p5.append(float(properties['sccEne'][()]))
            p6.append(float(properties['3rdEne'][()]))
            p7.append(float(properties['repEne'][()]))
            p8.append(float(properties['mbdEne'][()]))
            p9.append(np.linalg.norm(properties['TBdip'][()]))
            # electronic
            p10.append(np.array(properties['TBeig'][()]))
            # atomic
            p11.append(np.array(properties['TBchg'][()]))
            # target property (if present in dataset)
            if target_property in properties.keys():
                target.append(float(properties[target_property][()]))
    p11 = [pad_charges(x, configuration['max_atoms']) for x in p11]
    target = np.array(target) if len(target)>0 else np.array([0.0]*len(xyz))
    print(f"Number of conformers in dataset: {len(target)}")

    # standarization of QM properties
    _vars = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
    prop_array_list = []
    for var in _vars:
        var2 = np.array(var)
        try:
            __ = var2.shape[1]
        except IndexError:
            var2 = var2.reshape(-1, 1)
        scaler = StandardScaler()
        var3 = scaler.fit_transform(var2)
        prop_array_list.append(var3)

    # generate representation
    print(f"Generating {representation} descriptor")
    if representation=='props':
        descriptor = np.concatenate(prop_array_list, axis=1)
    elif representation=='bob':
        size, asize = configuration['max_atoms'], configuration['asize']
        descriptor = np.array([get_bob(Z[mol], xyz[mol], size, asize) for mol in range(len(target))])
    elif representation=='slatm':
        if 'mbtypes' not in configuration:
            mbs = get_slatm_mbtypes([Z[mol] for mol in range(len(target))])
            configuration.update(dict(mbtypes=mbs))
        else:
            mbs = configuration['mbtypes']
        descriptor = np.array([get_slatm(Z[mol], xyz[mol], mbs) for mol in range(len(target))])
    print(f"Add QM properties to {representation}?: {add_qm}")
    if add_qm and representation!='props':
        props = np.concatenate(prop_array_list, axis=1)
        descriptor = np.concatenate((descriptor, props), axis=1)
    print(f"Descriptor dimensions: {descriptor.shape}")

    return descriptor, target, configuration

def load_xgb(parameters):
    # load model from pickle file
    trained_model = parameters['estimator']
    return trained_model

def load_krr(parameters, training_dataset):
    assert len(training_dataset)>0 and os.path.exists(training_dataset), "The training dataset must be provided."
    # load hyperparameters from pickle file
    hyperparameters = json.loads(parameters['hyperparams'])

    # prepare training data
    X, y, _ = prepare_data(dataset_path=training_dataset, 
                           representation=parameters['representation'], 
                           add_qm=parameters['add_qm'], 
                           target_property=parameters['target_property'], 
                           configuration=parameters)
    train_idxs = parameters['X_train_idxs']
    X_train, y_train = X[train_idxs,:], y[train_idxs]
    # build KRR model
    estimator = OptimizedKRR(**hyperparameters)
    estimator.fit(X_train, y_train)
    return estimator

def evaluate_regressor(dataset_path, model_path, training_dataset, mode='validation'):
    # load model parameters from pickle file
    with open(model_path, "rb") as f:
        parameters = pickle.load(f)

    start = time.time()
    if parameters['architecture']=='xgboost':
        trained_model = load_xgb(parameters)
    elif parameters['architecture']=='krr':
        trained_model = load_krr(parameters, training_dataset)
    end = time.time()
    print(f"Loaded model: {parameters['architecture']}")
    print(f"Execution time: {end-start:.6f} seconds")

    # dataset over which do validation/inference
    start = time.time()
    X, y, configuration = prepare_data(dataset_path=dataset_path, 
                                       representation=parameters['representation'], 
                                       add_qm=parameters['add_qm'], 
                                       target_property=parameters['target_property'], 
                                       configuration=parameters)
    end = time.time()
    print(f"Execution time: {end-start:.6f} seconds")

    # validation: metrics of QUED paper
    if mode=='validation':
        test_idxs = parameters['X_test_idxs']
        X_test, y_test = X[test_idxs], y[test_idxs]
        y_pred = trained_model.predict(X_test).flatten()
        print(f"Validation of {parameters['target_property']} prediction:")
        for i in np.linspace(0, len(y_pred)-1, 10, dtype=int):
            print(f"True: {y_test[i]:.6f} | Prediction: {y_pred[i]:.6f}")
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae: .3f}")
        print(f"RMSE: {rmse: .3f}")
        print(f"R2: {r2: .3f}")
    # inference: on new data
    elif mode=='inference':
        X_test = X
        y_pred = trained_model.predict(X_test).flatten()
        print(f"Inference of {parameters['target_property']} prediction:")
        for i in range(len(y_pred)):
            print(f"Prediction: {y_pred[i]:.6f}")
    
    return X, trained_model, parameters

def shap_analysis(X, trained_model, parameters):
    assert parameters['architecture'] == 'xgboost', "SHAP analysis is only available for XGBoost"

    features = []
    if parameters['representation']=='props':
        features = qm_props_names[:-2]
        features += [f"{qm_props_names[-2]}_{i+1}" for i in range(8)]
        features += [f"{qm_props_names[-1]}_{i+1}" for i in range(parameters['max_atoms'])]
    else:
        features = [f"{parameters['representation']}_{i+1}" for i in range(X.shape[1])]
    if parameters['add_qm']:
        features += qm_props_names[:-2]
        features += [f"{qm_props_names[-2]}_{i+1}" for i in range(8)]
        features += [f"{qm_props_names[-1]}_{i+1}" for i in range(parameters['max_atoms'])]

    # Transform data with the scaler inside the pipeline
    X_std = trained_model['scaler'].transform(X)
    # SHAP analysis
    explainer = shap.TreeExplainer(trained_model['model'], feature_names=features)
    # explain the scaled test set
    shap_values = explainer(X_std)
    # Create the plot
    shap.plots.beeswarm(shap_values, max_display=12)
    # Save to file 
    sufix = f"{parameters['representation']}"
    if parameters['add_qm']: sufix += '_qm'
    plt.savefig(f'./shap-{sufix}.png', dpi=300, bbox_inches='tight')

    return shap_values

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dataset', type=str,
                        help='Relative or absolute path to the HDF5 file with conformers '
                             'geometries and QM properties. ')
    parser.add_argument('-v', '--validation', default=False, action='store_true',
                        help='Set mode to validation. '
                             'Default - False (inference). '
                             'Just use "-v" to corroborate MAE, RMSE and R2 reported. '
                             'In this case, use the corresponding training data as input_dataset. '
                             'Options: /path/to/ld50-stable.h5, /path/to/lipo-stable.h5, /path/to/qm7x-eq.h5, /path/to/qm7x-neq.h5'
                             'Otherwise dont use at all, it will predict property values using loaded model. ')
    parser.add_argument('-m', '--model_parameters', type=str, 
                        help='Path to pickle file with trained XGBoost or KRR model. '
                             'It contains estimator, descriptor type, target property, etc. ')
    parser.add_argument('-t', '--training_dataset', type=str, default='',
                        help='Only for KRR. '
                             'The training dataset can be found in the Github repository. '
                             'Options: /path/to/ld50-stable.h5, /path/to/lipo-stable.h5, /path/to/qm7x-eq.h5, /path/to/qm7x-neq.h5'
                             'It must be in HDF5 format. ')
    parser.add_argument('-sh', '--do_shap', default=False, action='store_true',
                        help='Perform SHAP analysis. '
                             'Default - False. '
                             'Only works with XGBoost models. '
                             'Recommended: use the corresponding training data as input_dataset. ') 
    return parser

if __name__ == '__main__':
    arguments = get_parser().parse_args()

    dataset_path = arguments.input_dataset
    model_path = arguments.model_parameters
    training_dataset = arguments.training_dataset
    mode = 'validation' if arguments.validation else 'inference'

    X, trained_model, parameters = evaluate_regressor(dataset_path, model_path, training_dataset, mode=mode)

    if arguments.do_shap:
        assert parameters['architecture'] == 'xgboost', "SHAP analysis is only available for XGBoost"
        shap_values = shap_analysis(X, trained_model, parameters)
