import os
from argparse import ArgumentParser

import numpy as np
import h5py

from ase.io import read
from qml.representations import generate_coulomb_matrix, generate_bob, generate_slatm, get_slatm_mbtypes
from krr_opt import *

# values for tox
# max_atoms: The maximum number of atoms in the representation
# asize: The maximum number of atoms of each element type supported by the representation
max_atoms = 196
asize_tox = {'C': 62, 'H': 111, 'N': 16, 'O': 35, 'S': 8, 'Cl': 12, 'F': 20, 'Br': 6, 'I':6, 'P': 4}
atomtypes = dict(asize_tox.keys())


def get_bob(filename, size=max_atoms, asize=asize_tox):
    atoms = read(filename)
    Z = atoms.get_atomic_numbers()
    xyz = atoms.get_positions()

    atomtypes = dict(asize.keys())
    bob_repr = generate_bob(Z, xyz, 
                            size=max_atoms, 
                            atomtypes=atomtypes, 
                            asize=asize)
    return bob_repr
