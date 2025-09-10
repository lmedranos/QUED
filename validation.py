"""
Evaluate XGBoost model
"""

import os
from argparse import ArgumentParser

import yaml
import numpy as np
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from electronic import *
from geometric import *

# values for tox
# max_atoms: The maximum number of atoms in the representation
# asize: The maximum number of atoms of each element type supported by the representation
max_atoms = 196
asize = {'C': 62, 'H': 111, 'N': 16, 'O': 35, 'S': 8, 'Cl': 12, 'F': 20, 'Br': 6, 'I':6, 'P': 4}
atomtypes = list(asize.keys())

def farthest_point_sampling(points, num):
    ###
    # Farthest Point Sampling (FPS) 
    # points: point cloud (dataframe)
    # num: number of wished selected points
    ###
    #points.index = range(0,len(points))

    center = np.mean(points, axis=0)            # Due to PCA, already shift to (0,0,0)
    fps_idx = np.zeros(num, dtype=np.int32)     # initiate the idx_list

    # initial points, largest distance from center
    # calculate the euclidean distance for every points
    distances = np.linalg.norm(points - center, axis=1)
    fps_idx[0] = np.argmax(distances)

    # calculate the distance between one point and selected_pt_group
    for i in range(1, num): 
        # calculate all the rest points to the selected new point
        current_points = points[fps_idx[i-1]]
        new_distances = np.linalg.norm(points - current_points,axis=1) 
        # calculate the shortest distance as the criteria for each point
        # make it more clear: take the point as criteria with the shortest distance 
        distances = np.minimum(distances, new_distances)
        # take the maximal among those distances, then it is the farthest point
        fps_idx[i] = np.argmax(distances)

    return fps_idx

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_conformers', type=str,
                        help='Relative or absolute path to xyz file of conformers '
                        'or the directory with the xyz files. ')
    return parser

arguments = get_parser().parse_args()

conformers_path = arguments.input_conformers
if conformers_path.endswith('xyz'):
    xyzFiles = [conformers_path]
else:
    xyzFiles = [os.path.join(conformers_path, x) for x in os.listdir(conformers_path)]

max_conformers = 10
bob_qm = []

#### starts from xyz file ####
for inputFile in xyzFiles:
    for i in range(max_conformers):
        Z, xyz = get_atomic_numbers_n_coordinates(inputFile, i)
        if Z.sum()==0: 
            print(f"{inputFile}: Maximum of conformers reached: {i} conformers")
            break

        #### get geometric descriptor ####
        _bob = generate_bob(Z, xyz, 
                            size=max_atoms, 
                            atomtypes=atomtypes, 
                            asize=asize)

        #### get electronic descriptor ####
        DFTBprops = calculate_dftb_props(Z, xyz)
        TBdip = np.array([np.linalg.norm(DFTBprops[8:11])])
        nEig = int(DFTBprops[11])
        TBeig = DFTBprops[12:12+nEig]
        TBchg = DFTBprops[12+nEig:]
        TBchg = pad_charges(TBchg, max_atoms)
        _props = np.concatenate((DFTBprops[:8], TBdip, TBeig, TBchg))
        
        rep = np.concatenate((_bob, _props))
        bob_qm.append(rep)
bob_qm = np.array(bob_qm)

#### build XGBoost model ####
dataset_path = '/data/horse/ws/alhi416g-qmbio2/tox/descriptor/representation-minenergy.npz'
dataset = np.load(dataset_path)
X, y = dataset['bob_qm'], dataset['ld50']
# farthest point sampling
tsne = TSNE(n_components=2, random_state=20240815, perplexity=30, max_iter=1000, n_jobs=-1)
X_tsne = tsne.fit_transform(X)
train_idxs = farthest_point_sampling(X_tsne, 5000)
test_idxs = np.array([i for i in range(len(y)) if i not in train_idxs])
X_train, y_train = X[train_idxs,:], y[train_idxs]
X_test, y_test   = X[test_idxs,:], y[test_idxs]

# load hyperparameters
xgbr = {"model": xgb.XGBRegressor(n_jobs=-1)}
hyperparameters_path = '/data/horse/ws/alhi416g-qmbio2/tox/xgb/minenergy/bob/all/5000/hyperpars.yaml'
with open(hyperparameters_path) as f:
    hyperparams = yaml.load(f, Loader=yaml.FullLoader)
trained_model = xgbr['model'].set_params(**hyperparams)

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('model', trained_model)])
pipeline.fit(X_train, y_train)

# validation / inference
y_pred = pipeline.predict(bob_qm)
k = 0
for inputFile in xyzFiles:
    for i in range(arguments.max_conformers):
        print(f'File {inputFile} | conformer {i}: Prediction = {y_pred: .3f}')