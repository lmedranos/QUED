# -*- coding: utf-8 -*-

import os
import gc
import shutil
import logging

import numpy as np
import schnetpack as spk
import matplotlib.pyplot as plt

from itertools import product
from timeit import default_timer
from platform import python_version
from argparse import ArgumentParser
from collections.abc import Iterable

from joblib import Parallel, delayed, cpu_count, parallel_config
from multiprocessing import set_start_method

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.preprocessing import StandardScaler
from qml.representations import generate_coulomb_matrix, \
    generate_bob, generate_slatm, get_slatm_mbtypes

from krr_opt import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s\n\r%(message)s',
    datefmt='%H:%M:%S'
)

logger.info(
    f'python version: {python_version()}, '
    f'number of cores: {cpu_count()}'
)

ALLOWED_PROPERTIES = (
    "RMSD", "EAT", "EMBD", "EGAP", "POL", "DIP", "KSE",
    "FermiEne", "BandEne", "NumElec", "h0Ene", "sccEne",
    "3rdEne", "RepEne", "mbdEne", "TBdip", "TBeig", "TBchg"
)

AVAILABLE_PROP_SET = ('all', 'global', 'electronic', 'atomic')


def _arg2tuple(arg):
    if not isinstance(arg, Iterable):
        return arg,
    return tuple(arg)


def _remove_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(
            f'following directory '
            f'has been created: {path}'
        )
    else:
        logger.warning(
            f'following directory {path} already '
            f'exists and will be removed'
        )
        shutil.rmtree(path)
        os.makedirs(path)
        logger.info(
            f'following directory '
            f'has been created: {path}'
        )

def newgap(KSE2):
    """
    #TODO refactor + doc
    """
    egdftb = []
    nmols = len(KSE2)
    print(nmols)
    for kk in range(nmols):
      egdftb.append(abs(KSE2[kk][3]-KSE2[kk][4]))

    return egdftb


def prepare_data(
        prop_name,
        dataset_path,
        cm=False,
        bob=False,
        slatm=False,
        props=False,
        add_qm_props=False,
        allowed_props=(),
        prop_set='all',
):
    """
    #TODO refactor + doc
    """
    def _complete_list(prop_list):
        tmp = []
        for prop in prop_list:
            element = np.concatenate(
                (prop, np.zeros((23 - len(prop)))),
                axis=None
            ) if len(prop) != 23 else prop
            tmp.append(element)

        return tmp

    # data preparation
    dataset = spk.data.AtomsData(dataset_path, load_only=allowed_props)
    n = len(dataset)

    logger.info(
        f'dataset size: {n}; '
        f'dataset path: {dataset_path}'
    )

    np.random.seed(2314)
    idxs = np.random.permutation(np.arange(n))

    logger.info(
        f'mol indices after permutation - '
        f'20 first: {idxs[:20]}, '
        f'20 last: {idxs[-20:]}'
    )

    # Computing predicted property
    logger.info('collecting properties...')

    xyz, Z, e_atom, e_gap, target = [], [], [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = \
        [], [], [], [], [], [], [], [], [], [], []
    for i in idxs:
        atoms, properties = dataset.get_properties(i)

        if not int(len(properties['KSE'])):
            continue

        xyz.append(atoms.get_positions())
        Z.append(atoms.get_atomic_numbers())
        e_atom.append(float(properties['EAT']))
        e_gap.append(float(properties['EGAP']))
        target.append(float(properties[prop_name]))

        # global
        p1.append(float(properties['FermiEne']))
        p2.append(float(properties['BandEne']))
        p3.append(float(properties['NumElec']))
        p4.append(float(properties['h0Ene']))
        p5.append(float(properties['sccEne']))
        p6.append(float(properties['3rdEne']))
        p7.append(float(properties['RepEne']))
        p8.append(float(properties['mbdEne']))
        p9.append(np.linalg.norm(properties['TBdip'].numpy()))
        # electronic
        p10.append(properties['TBeig'].numpy())
        # atomic
        p11.append(properties['TBchg'].numpy())

    p11 = _complete_list(p11)

    eg_dftb = newgap(p10)
    eg_dftb = np.abs(np.array(eg_dftb)).tolist()

    # Standardize the data property
    _vars = []
    _vars_names = []
    for p in prop_set:
        if p=='all': 
            _vars = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
            _vars_names = ['FermiEne', 'BandEne', 'NumElec', 'h0Ene', 'sccEne', '3rdEne', 'RepEne', 'mbdEne', 'TBdip', 'TBeig', 'TBchg']
        elif p=='global': 
            _vars += [p1, p2, p3, p4, p5, p6, p7, p8, p9, eg_dftb]
            _vars_names += ['FermiEne', 'BandEne', 'NumElec', 'h0Ene', 'sccEne', '3rdEne', 'RepEne', 'mbdEne', 'TBdip', 'EGAPdftb']
        elif p=='electronic': 
            _vars += [p10]
            _vars_names += ['TBeig']
        elif p=='atomic': 
            _vars += [p11]
            _vars_names += ['TBchg']
    logger.info(f'extracting {prop_set} properties: {_vars_names}')

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
    del _vars
    del _vars_names

    logger.info('collecting properties - done')

    # Generate representations
    logger.info(
        f'generating representations for '
        f'cm: {cm}, bob: {bob}, slatm: '
        f'{slatm}, props: {props} ...'
    )

    cm_repr, bob_repr, slatm_repr = np.array([]), np.array([]), np.array([])

    # Coulomb matrix
    if cm:
        cm_repr = np.array([generate_coulomb_matrix(
            Z[mol], xyz[mol], sorting='unsorted'
        ) for mol in range(len(target))])
    # Bag-of-bonds
    if bob:
        bob_repr = np.array([generate_bob(
            Z[mol], xyz[mol], atomtypes={'C', 'H', 'N', 'O', 'S', 'Cl'},
            asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1, 'Cl': 2}
        ) for mol in range(len(target))])
    # SLATM
    if slatm:
        mbs = get_slatm_mbtypes([Z[mol] for mol in range(len(target))])
        slatm_repr = np.array([generate_slatm(
            xyz[mol], Z[mol], mbs
        ) for mol in range(len(target))])

    logger.info('generating representations - done')

    cm_final, bob_final, slatm_final, props_final = np.array([]), np.array([]), np.array([]), np.array([])

    prop_vectors = np.concatenate(prop_array_list, axis=1)
    if props:
        props_final = prop_vectors

    if add_qm_props:
        if cm:
            cm_final = np.concatenate((cm_repr, prop_vectors), axis=1)
        if bob:
            bob_final = np.concatenate((bob_repr, prop_vectors), axis=1)
        if slatm:
            slatm_final = np.concatenate((slatm_repr, prop_vectors), axis=1)
    else:
        cm_final = cm_repr
        bob_final = bob_repr
        slatm_final = slatm_repr

    target = np.array(target)

    if cm:
        logger.info(f'descriptor dimensions: {cm_final.shape}')
    if bob:
        logger.info(f'descriptor dimensions: {bob_final.shape}')
    if slatm:
        logger.info(f'descriptor dimensions: {slatm_final.shape}')

    return dict(
        cm=cm_final,
        bob=bob_final,
        slatm=slatm_final,
        props=props_final,
        target_props=target,
        idxs_to_permute=range(len(target))
    )


def _perform_random_iter(
        n_iter=None,
        n_train=None,
        pidx=None,
        n_workers_opt=None,
        X1=None,
        X2=None,
        y=None,
        n_val=None,
        reg_params=None,
        estimator_class=None,
        single_krr=None,
        baseline=None,
        kernels=None,
        p_norms=None,
        gen_powers=None,
        metrics=None,
        reprs=None
):
    prefix = f"n_train={n_train}; n_val={n_val}; n_iter={n_iter}"
    logger.info(
        f'data preparation for {prefix}; '
        f'first 20 idxs of the current permutation: '
        f'{pidx[:20]}; last 20: {pidx[-20:]}'
    )

    # data preparation
    target, repr1 = y[pidx], X1[pidx]
    XA_train, XA_val, XA_test = repr1[:n_train], repr1[-n_val:], repr1[n_train:-n_val]
    y_train, y_val, y_test = target[:n_train], target[-n_val:], target[n_train:-n_val]
    X_train, X_val, X_test = (XA_train, ), (XA_val, ), (XA_test, )

    del repr1
    del target
    gc.collect()

    if X2 is not None:
        repr2 = X2[pidx]
        XB_train, XB_val, XB_test = repr2[:n_train], repr2[-n_val:], repr2[n_train:-n_val]
        X_train, X_val, X_test = (XA_train, XB_train), (XA_val, XB_val), (XA_test, XB_test)

        del repr2
        gc.collect()

    logger.info(
        f'min target prop values for train/val/test: '
        f'{y_train.min(), y_val.min(), y_test.min()}; '
        f'max target prop values for train/val/test: '
        f'{y_train.max(), y_val.max(), y_test.max()}; '
        f'{prefix}: data preparation - done'
    )

    prefix = f"n_train={n_train}; n_val={n_val}; reg={reg_params}"
    logger.info(
        f'{prefix}: '
        f'fitting estimator in parallel for '
        f'{n_train} training pts and '
        f'{reg_params} regularization param(s)'
    )

    estimator = estimator_class()

    if single_krr or baseline:
        params = {'alpha': reg_params[0]}
    else:
        params = {'alpha': reg_params[0], 'beta': reg_params[1]}

    if len(reprs) == 1:
        params.update({'ker1_type': kernels[0],
                       'ker1_pow': gen_powers[0],
                       'ker1_metric': metrics[0],
                       'ker1_metric_kwargs': {'p': p_norms[0]}})
    else:
        params.update({'ker1_type': kernels[0], 'ker2_type': kernels[1],
                       'ker1_pow': gen_powers[0], 'ker2_pow': gen_powers[1],
                       'ker1_metric': metrics[0], 'ker1_metric_kwargs': {'p': p_norms[0]},
                       'ker2_metric': metrics[1], 'ker2_metric_kwargs': {'p': p_norms[1]}})

    estimator.set_params(**params)
    estimator.fit(*X_train, y_train)

    del X_train
    gc.collect()

    yp_val = estimator.optimize(*X_val, y_val, n_workers_opt=n_workers_opt)
    rmse_val = mean_squared_error(y_val, yp_val, squared=False)
    mae_val = mean_absolute_error(y_val, yp_val)

    logger.info(
        f'optimization for {prefix} is done '
        f'VAL - RMSE: {rmse_val}, MAE: {mae_val},'
        f'best sigmas: {estimator.get_sigmas}, '
        f'best regularizers: {estimator.get_reg_params}'
    )

    # Predictions
    # No train/val predictions with optimized parameters
    # are made to save some memory, they can be easily
    # obtained using the information from saved dump
    yp_test = estimator.predict(*X_test)

    rmse_test = mean_squared_error(y_test, yp_test, squared=False)
    mae_test = mean_absolute_error(y_test, yp_test)

    params_dict = estimator.get_params()
    logger.info(
        f'{prefix}: TEST - RMSE: {rmse_test}; MAE: '
        f'{mae_test}, estimator best params: {params_dict}'
    )

    data_dict = dict(
        weights=estimator.get_weights,
        hyperparams=params_dict,
        X_train_idxs=pidx[:n_train],
        X_val_idxs=pidx[-n_val:],
        X_test_idxs=pidx[n_train:-n_val],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        val_predictions=yp_val,
        test_predictions=yp_test
    )

    return dict(
        rmse_val=rmse_val,
        mae_val=mae_val,
        rmse_test=rmse_test,
        mae_test=mae_test
    ), data_dict


def _train_parallel(
        n_train=None,
        n_iters=None,
        n_cpus_inner=None,
        n_workers_opt=None,
        X1=None,
        X2=None,
        y=None,
        n_val=None,
        results_dir=None,
        dataset_name=None,
        reg_params=None,
        idxs_to_permute=None,
        estimator_class=None,
        y_min=None,
        y_max=None,
        single_krr=None,
        baseline=None,
        kernels=None,
        p_norms=None,
        gen_powers=None,
        metrics=None,
        reprs=None
):
    """
    #TODO refactor + doc
    """
    logger.info(
        f'Parallel training job is in progress '
        f'for n_train={n_train}, n_iters={n_iters}, '
        f'n_cpus_inner={n_cpus_inner}'
    )

    np.random.seed(n_iters)
    perms_list = [
        np.random.permutation(idxs_to_permute)
        for __ in range(n_iters)
    ]

    with parallel_config(backend='loky', inner_max_num_threads=n_workers_opt):

        random_results_list = Parallel(n_jobs=n_cpus_inner)(
            delayed(_perform_random_iter)(
                n_iter=i,
                n_train=n_train,
                pidx=pidx,
                n_workers_opt=n_workers_opt,
                X1=X1,
                X2=X2,
                y=y,
                n_val=n_val,
                reg_params=reg_params,
                estimator_class=estimator_class,
                single_krr=single_krr,
                baseline=baseline,
                kernels=kernels,
                p_norms=p_norms,
                gen_powers=gen_powers,
                metrics=metrics,
                reprs=reprs
            ) for i, pidx in
            zip(range(n_iters), perms_list)
        )

    sorted_val_res_list = sorted(random_results_list, key=lambda x: x[0]['mae_val'])

    maes_val, mses_val, maes_test, mses_test = list(zip(*[
        (x[0]['mae_val'], x[0]['rmse_val'], x[0]['mae_test'], x[0]['rmse_test'])
        for x in sorted_val_res_list
    ]))

    best_metrics, best_params = sorted_val_res_list[0]

    np.savez_compressed(
        f'{results_dir}/dumps/{n_train}pts.npz',
        **best_params
    )

    return best_metrics


def main_qm_dataset(
        dataset_path=None,
        reprs=('cm', 'slatm'),
        kernels=('gau', 'gau'),
        metrics=('euclidean', 'euclidean'),
        p_norms=(2.0, 2.0),
        gen_powers=(None, None),
        val_points=(500, ),
        train_points=(500, ),
        sampling_iters=(1, ),
        results_dir='./current_results',
        target_prop='IntE1',
        add_qm_props=True,
        reg_params=(1e-3, ),
        baseline=False,
        n_cpus_outer=1,
        n_cpus_inner=(1, ),
        n_workers_opt=1, 
        allowed_props=ALLOWED_PROPERTIES,
        to_kcal_mol=True,
        prop_set='all'
):
    """
    #TODO refactor + doc
    """
    n_kernels = len(reprs)
    single_krr = True \
        if n_kernels == 1 else False
    if baseline:
        estimator_class = BaselineKRR
    elif single_krr:
        estimator_class = SingleKRR
    else:
        estimator_class = DoubleKRR

    dataset_name = f'{dataset_path.split("/")[-1]}_' \
                   f'{str(train_points)}_subsets_' \
                   f'{n_kernels}_ker_on_{str(reprs)}_' \
                   f'{str(kernels)}_{str(metrics)}_{str(p_norms)}_' \
                   f'qm_props_{str(add_qm_props)}_' \
                   f'target_{target_prop}_out_{n_cpus_outer}_' \
                   f'in_{n_cpus_inner}_in_w_{n_workers_opt}'
    _remove_create_dir(f'{results_dir}/dumps/')

    reprs_to_generate = {
        'cm': False, 'bob': False,
        'slatm': False, 'props': False
    }
    reprs_to_generate.update(
        {r: True for r in reprs}
    )
    logger.info(
        f'descriptors order - '
        f'{[r for r in reprs if reprs_to_generate[r]]}'
    )

    descriptor_data = prepare_data(
        target_prop,
        dataset_path,
        cm=reprs_to_generate['cm'],
        bob=reprs_to_generate['bob'],
        slatm=reprs_to_generate['slatm'],
        props=reprs_to_generate['props'],
        add_qm_props=add_qm_props,
        allowed_props=allowed_props,
        prop_set=prop_set
    )

    X = [descriptor_data[r] for r in reprs if reprs_to_generate[r]]
    X1, X2 = X[0], X[1] if n_kernels == 2 else None
    y, idxs_to_permute = descriptor_data['target_props'], \
                         descriptor_data['idxs_to_permute']
    del X
    del descriptor_data
    gc.collect()

    if to_kcal_mol:
        y = 23.06 * y  # 1 eV is 23.06 kcal/mol
    y_min, y_max = y.min(), y.max()

    logger.info(
        f'target props min; max: {y_min, y_max}; '
        f'reg params: {reg_params}; '
        f'CPUs outer loop: {n_cpus_outer}; '
        f'CPUs inner loop: {n_cpus_inner}; '
        f'workers inner loop: {n_workers_opt}'
    )

    mse_val, mae_val, mse_test, mae_test = [], [], [], []

    res_list = Parallel(n_jobs=n_cpus_outer)(
        delayed(_train_parallel)(
            n_train=nt,
            n_iters=si,
            n_cpus_inner=ci,
            n_workers_opt=n_workers_opt,
            X1=X1,
            X2=X2,
            y=y,
            n_val=nv,
            results_dir=results_dir,
            dataset_name=dataset_name,
            reg_params=reg_params,
            idxs_to_permute=idxs_to_permute,
            estimator_class=estimator_class,
            y_min=y_min,
            y_max=y_max,
            single_krr=single_krr,
            baseline=baseline,
            kernels=kernels,
            p_norms=p_norms,
            gen_powers=gen_powers,
            metrics=metrics,
            reprs=reprs
        ) for nt, nv, si, ci in zip(
            train_points, val_points,
            sampling_iters, n_cpus_inner
        )
    )

    for res in res_list:
        mse_val.append(res['rmse_val'])
        mae_val.append(res['mae_val'])
        mse_test.append(res['rmse_test'])
        mae_test.append(res['mae_test'])


    performance_dict = dict(
        train_points=train_points,
        mse_val=mse_val,
        mae_val=mae_val,
        mse_test=mse_test,
        mae_test=mae_test
    )
    np.savez_compressed(
        f'{results_dir}/dumps/performance.npz',
        **performance_dict
    )


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dataset_path', type=str,
                        help='Relative or absolute path to the initial dataset. '
                             'String type. '
                             'Examples: '
                             '1) -i "./DATASET" - relative '
                             '2) -i "/PATH/TO/DATASET" - absolute')
    parser.add_argument('-r', '--representations', nargs='+', type=str, default=('cm', 'slatm'),
                        help='Listed representations (case-insensitive) to be used. '
                             'String type each, default - "cm" "slatm". '
                             'Specify 1 representation to perform single kernel optimization and 2 for double kernel; '
                             '"cm", "bob", "slatm" and "props" are currently available. '
                             'Examples: '
                             '1) -r "cm" - use single kernel on Coulomb matrix '
                             '2) -r "cm" "slatm" - use double kernel on Coulomb matrix and SLATM '
                             '3) -r "cm" "props" - use double kernel on Coulomb matrix and properties')
    parser.add_argument('-k', '--kernels', nargs='+', type=str, default=('gau', 'gau'),
                        help='Listed kernel functionals (case-sensitive) to be used for kernel 1 and 2 respectively. '
                             'String type each, default - "gau" "gau". '
                             'Specify 1 functional if you have single kernel optimization and 2 for double kernel; '
                             f'Must be of of the following: {str(AVAILABLE_KERNELS)}')
    parser.add_argument('-m', '--metrics', nargs='+', type=str, default=('euclidean', 'euclidean'),
                        help='Listed metric distances (case-sensitive) to be used for kernel 1 and 2 respectively. '
                             'String type each, default - "euclidean", "euclidean". '
                             'Specify 1 metric if you have single kernel optimization and 2 for double kernel; '
                             f'Must be of of the following: {str(_VALID_METRICS)}')
    parser.add_argument('-n', '--norms', nargs='+', type=float, default=(2, 2),
                        help='Listed p_norms for Minkowski distances to be used for kernel 1 and 2 respectively. '
                             'Float type each, default - 2.0 2.0. '
                             'Specify only if you use minkowski distance for any of the kernels.'
                             f'Can be from 0 to 1 as well as from 1 to infinity.')
    parser.add_argument('-v', '--val_frac', nargs='+', type=float, default=1.0,
                        help='Listed fractions (f), such that f = n_validation / n_train. Must be of '
                             'equal length to number of training subsets. Float type each, default - 1.0. '
                             'Example: '
                             '-t 100 200 -v 1.0 0.75 - use 100 val pts for 100 '
                             'training pts and 150 val pts for 200 training pts.')
    parser.add_argument('-t', '--train_points', nargs='+', type=int, default=500,
                        help='Listed numbers of training points used for modeling. '
                             'Integer type each, default - 500. '
                             'Example: '
                             '-t 100 200 300 400 - build models for 100, 200, 300 and 400 training pts')
    parser.add_argument('-s', '--sampling_iters', nargs='+', type=int, default=1,
                        help='Listed numbers of iterations used to find best model for each training subset. '
                             'Integer type each, default - 1. '
                             'Example: '
                             '-it 2 4 -t 100 200 - sample 2 times for 100 training pts and 4 times for 400.')
    parser.add_argument('-o', '--output_path', type=str, default='./current_results',
                        help='Path to output files directory. '
                             'Sting type, default - "./current_results". '
                             'All the output will be put into this directory, '
                             'subdirectories for different experiments are created automatically.')
    parser.add_argument('-p', '--params_reg', nargs='+', type=float, default=1e-3,
                        help='Initial regularization parameter. '
                             'Float type, default - 1e-3. '
                             'Example: '
                             '-p 1e-1 - find best model with 1e-1 as starting reg. parameter for single kernel, '
                             'in case of the double kernel approach two values are expected, they will be set as '
                             'starting regularization parameters and will be further optimized anyway.')
    parser.add_argument('-q', '--add_qm', default=False, action='store_true',
                        help='Concatenate qm props with descriptor (!) representations. '
                             'Default - False. '
                             'Just use "-q" to add these properties to your descriptor, otherwise dont use at all.')
    parser.add_argument('-b', '--get_baseline', default=False, action='store_true',
                        help='Get baseline model for 2-kernel (!) representations only. '
                             'Default - False. '
                             'Use "-b" to perform baseline calculations for 2-kernel, otherwise dont use this flag.')
    parser.add_argument('-y', '--target_property', type=str, default='EAT',
                        help=f'Property to predict (case-insensitive). '
                             f'Sting type, default - "EAT". '
                             f'Must be of of the following: {str(ALLOWED_PROPERTIES)}')
    parser.add_argument('-co', '--cpus_outer', type=int, default=1,
                        help='Number of CPUs allocated per outer loop (over train subsets). '
                             'Integer type, default - 1 (for sequential outer loop).')
    parser.add_argument('-ci', '--cpus_inner', nargs='+', type=int, default=1,
                        help='Number of CPUs allocated per inner loop (over # of iterations). '
                             'Integer or iterable type, default - 1 (for sequential inner loop). '
                             'If iterable, then the length must be equal to the length of sampling iters (-s key).')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of workers used per inner loop CPU (for optimization). '
                             'Integer type, default - 1.')
    parser.add_argument('-gp', '--gen_powers', nargs='+', type=float, default=(None, None),
                        help='Listed powers only for the generic kernel ("gen") to be used for kernel 1 and 2 respectively.'
                             'Float type each, default - None None. ')
    parser.add_argument('-ps', '--prop_set', nargs='+', type=str, default='all',
                        help='Subset of electronic properties that will be included in the QM descriptor. '
                             'String type each, default - "all" (all sub sets). '
                             f"Must be of of the following: {str(AVAILABLE_PROP_SET)}")
    return parser


if __name__ == '__main__':
    set_start_method('spawn', force=True)  # pythonspeed.com/articles/python-multiprocessing/

    arguments = get_parser().parse_args()

    # general arguments preprocessing
    reprs = tuple([x.lower() for x in _arg2tuple(arguments.representations)])
    target_prop = arguments.target_property
    kernels = _arg2tuple(arguments.kernels)
    metrics = _arg2tuple(arguments.metrics)
    prop_set = _arg2tuple(arguments.prop_set)
    assert 1 <= len(reprs) <= 2, 'currently only single and double kernels are available'
    assert set(reprs).issubset({'cm', 'slatm', 'bob', 'props'}), 'wrong representation(s) used'
    assert target_prop.lower() in [x.lower() for x in ALLOWED_PROPERTIES], 'wrong target property'
    assert len(reprs) == len(kernels) == len(metrics), \
        'check number of kernels, reprs or metrics depending on whether you want to perform 1-KRR or 2-KRR'

    # points-related arguments preprocessing
    train_points = _arg2tuple(arguments.train_points)
    val_frac = _arg2tuple(arguments.val_frac)
    assert len(val_frac) == len(train_points), 'Numbers of validation and train subsets are different'
    val_points = tuple(int(f * t) for f, t in zip(val_frac, train_points))

    # sampling-related arguments preprocessing
    n_cpus_inner = _arg2tuple(arguments.cpus_inner)
    sampling_iters = _arg2tuple(arguments.sampling_iters)
    # TODO write the following in a more intuitive way
    if len(n_cpus_inner) == 1 and n_cpus_inner[0] == 1:
        # setting 1 cpu per sampling iteration in the default case
        n_cpus_inner = sampling_iters
    else:
        # some value of inner cpus must be assigned
        # to each set of sampling iterations
        assert len(n_cpus_inner) == len(sampling_iters)

    input_parameters = dict(
        dataset_path=arguments.input_dataset_path,
        reprs=reprs,
        kernels=kernels,
        metrics=metrics,
        p_norms=_arg2tuple(arguments.norms),
        gen_powers=_arg2tuple(arguments.gen_powers),
        val_points=val_points,
        train_points=train_points,
        sampling_iters=sampling_iters,
        results_dir=arguments.output_path,
        target_prop=target_prop,
        add_qm_props=arguments.add_qm,
        reg_params=_arg2tuple(arguments.params_reg),
        baseline=arguments.get_baseline,
        n_cpus_outer=arguments.cpus_outer,
        n_cpus_inner=n_cpus_inner,
        n_workers_opt=arguments.workers,
        prop_set=arguments.prop_set
    )

    logger.info(f'current named arguments: {input_parameters}')
    t1 = default_timer()

    main_qm_dataset(**input_parameters)

    t2 = default_timer()
    logger.info(f'total time: {t2 - t1}')