"""
Evaluate XGBoost model
"""

import os
import logging
from argparse import ArgumentParser

import h5py
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from krr_opt import *

from electronic import *
from geometric import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s\n\r%(message)s\n',
    datefmt='%H:%M:%S'
)

def get_trained_xgboost_model(dataset_path, model_path, representation, target):
    # load training dataset
    logger.info(f"loading data from {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)
    logger.info(f"extracting {representation} and {target}")
    X, y = dataset[representation], dataset[target]
    logger.info(f"dimensions of descriptor {representation}: {X.shape}")
    logger.info(f"loading data - done")

    # load model hyperparameters and train indices
    logger.info(f"loading parameters from {model_path}")
    parameters = np.load(model_path, allow_pickle=True)
    hyperparameters = parameters['hyperparams'].item()
    logger.info(f"loaded hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f'{key}: {value}')
    train_idxs = parameters['X_train_idxs']
    logger.info(f"loaded training indices: {train_idxs[:20]} ... {train_idxs[-20:]}")
    X_train, y_train = X[train_idxs,:], y[train_idxs]
    logger.info(f"number of training points: {X_train.shape}")
    logger.info(f"loading parameters - done")

    # build model
    logger.info('fitting XGBoost model')
    xgbr = {"model": xgb.XGBRegressor(n_jobs=-1)}
    trained_model = xgbr['model'].set_params(**hyperparameters)
    pipeline = Pipeline([('scaler', StandardScaler()),
                        ('model', trained_model)])
    pipeline.fit(X_train, y_train)
    logger.info('fitting XGBoost model - done')

    return pipeline

def get_trained_krr_model(dataset_path, model_path, representation, target):
    # load training dataset
    logger.info(f"loading data from {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)
    logger.info(f"extracting {representation} and {target}")
    X, y = dataset[representation], dataset[target]
    logger.info(f"dimensions of descriptor {representation}: {X.shape}")
    logger.info(f"loading data - done")

    # load model hyperparameters and train indices
    logger.info(f"loading parameters from {model_path}")
    parameters = np.load(model_path, allow_pickle=True)
    hyperparameters = parameters['hyperparams'].item()
    logger.info(f"loaded hyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f'{key}: {value}')
    weights = parameters['weights']
    logger.info(f"loaded weights: {weights[:20]} ... {weights[-20:]}")
    train_idxs = parameters['X_train_idxs']
    logger.info(f"loaded training indices: {train_idxs[:20]} ... {train_idxs[-20:]}")
    X_train, y_train = X[train_idxs,:], y[train_idxs]
    logger.info(f"number of training points: {X_train.shape}")
    logger.info(f"loading parameters - done")

    # build model
    logger.info('fitting KRR model')
    estimator = OptimizedKRR(**hyperparameters)
    estimator._weights = weights
    estimator._is_fitted = True
    estimator.X_train = X_train
    logger.info('fitting KRR model - done')

    return estimator

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_conformers', type=str,
                        help='Relative or absolute path to xyz file of conformers '
                             'or the directory with the xyz files. ')
    parser.add_argument('-n', '--max_conformers', type=int, default=10,
                        help='Maximum number of conformers per molecule to consider. ' 
                        'Defult: 10. ')
    parser.add_argument('-r', '--representation', type=str, default='props',
                        help='Representation (case-sensitive) to be used. '
                             'String type, default - "props". '
                             '"props" and "bob" are currently available. '
                             'Example: '
                             '-r "bob" - use xgboost with BOB ')
    parser.add_argument('-q', '--add_qm', default=False, action='store_true',
                        help='Concatenate qm props with descriptor (!) representations. '
                             'Default - False. '
                             'Just use "-q" to add these properties to your descriptor, otherwise dont use at all.')
    parser.add_argument('-t', '--training_dataset', type=str, 
                        help='Path to HDF5 file with training dataset.')
    parser.add_argument('-y', '--target_property', type=str, default='ld50',
                        help=f'Property to predict (case-sensitive). '
                             f'String type, default - "ld50". '
                             '"ld50" and "lipophilicity" are currently available. ')
    parser.add_argument('-a', '--model_architecture', type=str, default='xgboost',
                        help=f'ML model type (case-insensitive).'
                             f'String type, default - "xgboost". '
                             f'"krr" and "xgboost" are currently available. ')       
    parser.add_argument('-p', '--model_parameters', type=str, 
                        help='Path to npz file with parameters of trained ML model.')
    return parser

if __name__ == '__main__':
    arguments = get_parser().parse_args()

    #### generate conformers for input geometries ####
    logger.info(f"generate conformers for geometries in {arguments.input_conformers}")
    conformers_path = arguments.input_conformers
    if conformers_path.endswith('xyz'):
        xyzFiles = [conformers_path]
    else:
        xyzFiles = [os.path.join(conformers_path, x) for x in os.listdir(conformers_path)]
    max_conformers = arguments.max_conformers
    logger.info(f"up to {max_conformers} conformers will be extracted from each of the {len(xyzFiles)} files in {conformers_path}")

    logger.info(f"extract geometries")
    # get configuration of database
    if arguments.target_property=='ld50':
        size, asize = tox_database_config.values()
    elif arguments.target_property=='lipophilicity':
        size, asize = lipo_database_config.values()
    Z, xyz = [], [] 
    cids = []
    for inputFile in xyzFiles:
        for i in range(max_conformers):
            # get atomic numbers and coordinates
            _Z, _xyz = get_atomic_numbers_n_coordinates(inputFile, i)
            if _xyz.sum()==0: break
            Z.append(_Z)
            xyz.append(_xyz)
            cids.append(i)
    logger.info(f"number of conformers: {len(cids)}")
    logger.info(f"extract geometries - done")

    logger.info(f"generate {arguments.representation} descriptor")
    # generate qm descriptor
    if arguments.representation=='props':
        descriptor = np.array([get_props(Z[mol], xyz[mol], size) for mol in range(len(Z))])
    # generate bob descriptor
    elif arguments.representation=='bob':
        descriptor = np.array([get_bob(Z[mol], xyz[mol], size, asize) for mol in range(len(Z))])
    logger.info(f"generate {arguments.representation} descriptor - done")
    # add qm properties to geometric descriptor
    if arguments.add_qm and arguments.representation!='props':
        logger.info(f"add QM properties")
        props = np.array([get_props(Z[mol], xyz[mol], size) for mol in range(len(Z))])
        descriptor = np.concatenate((descriptor, props), axis=1)
        logger.info(f"Add QM properties - done")
    logger.info(f"dimensions of descriptor = {descriptor.shape}")

    #### build ML regressor model ####
    rep = arguments.representation if not arguments.add_qm else f"{arguments.representation}_qm"
    logger.info(f"load {arguments.model_architecture.upper()} model with {rep} descriptor")
    parameters = dict(dataset_path=arguments.training_dataset, 
                      model_path=arguments.model_parameters, 
                      representation=rep, 
                      target=arguments.target_property)
    if arguments.model_architecture.lower()=='xgboost':
        trained_model = get_trained_xgboost_model(**parameters)
    elif arguments.model_architecture.lower()=='krr':
        trained_model = get_trained_krr_model(**parameters)
    logger.info(f"load {arguments.model_architecture.upper()} model - done")

    # validation / inference
    logger.info("\nVALIDATION")
    y_pred = trained_model.predict(descriptor)
    for i, inputFile in enumerate(xyzFiles):
        logger.info(f'File {inputFile} | conformer {cids[i]}: Prediction = {y_pred[i]: .3f}')