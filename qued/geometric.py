import os
from argparse import ArgumentParser

import numpy as np
import h5py

from ase.io import read
from qml.representations import generate_bob, generate_slatm, get_slatm_mbtypes
from dscribe.descriptors import SOAP

# FIXED CONFIGURATION FOR TDCOMMONS-LD50 AND MOLECULENET-LIPOPHILICITY DATABASES
# max_atoms: The maximum number of atoms in the representation
# asize: The maximum number of atoms of each element type supported by the representation

qm7x_database_config = dict(max_atoms=23,
                            asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1, 'Cl': 2})

tox_database_config = dict(max_atoms=196,
                           asize={'C': 62, 'H': 111, 'N': 16, 'O': 35, 'S': 8, 'Cl': 12, 'F': 20, 'Br': 6, 'I':6, 'P': 4})

lipo_database_config = dict(max_atoms=203,
                            asize={'C': 66, 'H': 103, 'N': 17, 'O': 16, 'S': 3, 'Cl': 3, 'F': 9, 'Br': 2, 'I':3})

soap_config = dict(periodic=False, r_cut=6.0, n_max=8, l_max=6, average="outer")

def get_bob(Z, xyz, max_atoms, asize):
    atomtypes = set(asize.keys())
    bob_repr = generate_bob(Z, xyz, 
                            size=max_atoms, 
                            atomtypes=atomtypes, 
                            asize=asize)
    return bob_repr

def get_slatm(Z, xyz, mbtypes):
    slatm_repr = generate_slatm(xyz, Z, 
                                mbtypes=mbtypes)
    return slatm_repr

