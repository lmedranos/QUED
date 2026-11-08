# requirements
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s\n',
    datefmt='%H:%M:%S'
)

ENERGY_TERMS = {'dftb':  ['h0Ene', 'sccEne', '3rdEne', 'repEne', 'mbdEne'],
                'skala': ['el1Ene', 'el2Ene', 'repEne', 'xcEne', 'disEne'],
                'xtb':   ['xTBEnergy']}

@dataclass
class QUEDConformer:
    molecule_index: int
    conformer_index: int
    atomic_numbers: Optional[np.ndarray] = None
    atomic_coordinates: Optional[np.ndarray] = None
    properties: Optional[Dict[str, Any]] = None

@dataclass
class QUEDMolecule:
    index: int
    smiles: str
    target: float
    n_conformers: int

class QUEDDataset():
    def __init__(self, 
                 csv_path, 
                 hdf5_path, 
                 smiles_label: str = 'smile',
                 target_label: str = 'target',
                 id_label: Optional[str] = 'molID'):
        """
        Parameters
        ----------
        csv_path : str or Path
            Path to CSV file with molecule-level data (smiles, target values)
        hdf5_path : str or Path
            Path to HDF5 file with conformer-level data (atomic coordinates, QM properties)
        smiles_label : str, optional. Default: smile
        target_label : str, optional. Default: target
        id_label : str, optional. Default: molID
            Column names in CSV file to specify SMILE, target property, and molecule index
        """

        # Record database filepaths
        self.csv_path = Path(csv_path)
        self.hdf5_path = Path(hdf5_path)
        self.target_label = target_label
        
        # Read CSV to get SMILES
        self.df = pd.read_csv(self.csv_path)
        if id_label in self.df.columns:
            self.ids = self.df[id_label].astype(str).tolist()
        else:
            self.ids = [str(i) for i in range(len(self.df))]
        if smiles_label in self.df.columns:
            self.smiles = self.df[smiles_label].tolist()
        else:
            self.smiles = [None]
        self.target_values = self.df[self.target_label].to_numpy()

        # Temporal storing variables
        self._h5: Optional[h5py.File] = None
        self._n_conf_cache: Dict[int, int] = {}
        # Filter failed molecules (i.e., molecules without HDF5 entries)
        # keeping everything index-aligned
        self._filter_to_available()

    def _filter_to_available(self):
        """ 
        Drops CSV rows whose id has no corresponding 
        top-level (molecule-level) group in the HDF5 file. 
        """

        # Open the file briefly
        self._ensure_open()
        available = set(self._h5.keys())

        # Filter out molecules and keep a record
        keep_mask = [mol_id in available for mol_id in self.ids]
        n_dropped = len(keep_mask) - sum(keep_mask)
        self.missing_ids = [mid for mid, keep in zip(self.ids, keep_mask) if not keep]
        self.df = self.df.loc[keep_mask].reset_index(drop=True)
        self.ids = [mid for mid, keep in zip(self.ids, keep_mask) if keep]
        self.smiles = [smi for smi, keep in zip(self.smiles, keep_mask) if keep]
        self.target_values = self.target_values[np.array(keep_mask, dtype=bool)]

        if n_dropped>0:
            logger.info(f"QUEDDataset: dropped {n_dropped} / {len(keep_mask)} entries "
                        f"with no matching HDF5 group (e.g. failed conformer/QM jobs).")
        # O(1) reverse lookup: db id -> position in this (filtered) dataset.
        self._id_to_index: Dict[str, int] = {mid: i for i, mid in enumerate(self.ids)}

    # ---------- lifecycle ---------- 
    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # for context manager protocol (use dataset w a with statement)
    def __enter__(self):
        self._ensure_open()
        return self
 
    def __exit__(self, *exc):
        self.close()
 
    def __len__(self):
        return len(self.df)

    # ---------- molecule-level ----------
    def __getitem__(self, idx: int) -> QUEDMolecule:
        return QUEDMolecule(index=idx,
                            smiles=self.smiles[idx],
                            target=self.target_values[idx],
                            n_conformers=self._get_n_conformers(idx))

    def _get_n_conformers(self, idx: int) -> int:
        if idx in self._n_conf_cache:
            return self._n_conf_cache[idx]
        self._ensure_open()
        molecule = self._h5[self.ids[idx]]
        # Hierarchy: Molecule / Conformer -> count conformers
        n = len(molecule.keys())
        self._n_conf_cache[idx] = n
        return n

    def _get_conformer_keys(self, idx: int) -> List[str]:
        self._ensure_open()
        conformers = self._h5[self.ids[idx]]
        # numeric-aware sort
        return sorted(conformers.keys(), key=lambda k: (len(k), k))

    def get_mol(self, idx: int, add_hs: bool = False) -> Chem.Mol:
        # RDKit Mol built purely from SMILES
        mol = Chem.MolFromSmiles(self.smiles[idx])
        if add_hs and mol is not None:
            mol = Chem.AddHs(mol)
        return mol

    # ---------- conformer-level ----------
    def _h5_to_dict(self, group):
        # Recursively convert an HDF5 group into a Python dictionary
        return {
            key: self._h5_to_dict(value) if isinstance(value, h5py.Group)
            else value[()]
            for key, value in group.items()
        }

    def get_conformer(self, idx: int, conf_idx: int = 0) -> QUEDConformer:
        # Fetch a single stored (QM) conformer's data

        self._ensure_open()
        mol_id = self.ids[idx]
        keys = self._get_conformer_keys(idx)
        conf_key = keys[conf_idx]
        properties = self._h5[mol_id][conf_key]

        # Extract Geometry
        Z = properties['geometry']['atomic_numbers'][()]
        xyz = properties['geometry']['atomic_coordinates'][()]

        # Extract Properties (e.g., DFTB, Skala)
        props = {
            key: self._h5_to_dict(value)
            for key, value in properties.items()
            if key != 'geometry'
        }

        return QUEDConformer(molecule_index=idx,
                             conformer_index=conf_idx, 
                             atomic_numbers=Z,
                             atomic_coordinates=xyz,
                             properties=props)

    def iter_conformers(self, idx: int):
        # Generator over a molecule's stored conformers, one at a time
        for conf_idx in range(self._get_n_conformers(idx)):
            yield self.get_conformer(idx, conf_idx)

    def get_all_conformers(self, idx: int) -> List[QUEDConformer]:
        # Load every conformer for one molecule
        return list(self.iter_conformers(idx))

    def _conformer_energy(self, idx, conf_key, modal):
        # Calculate energy of a conformer
        grp = self._h5[self.ids[idx]][conf_key][modal]
        return sum(grp[k][()] for k in ENERGY_TERMS[modal])   # KeyError if a term is missing

    def get_lowest_energy_conformer(self, idx:int, modal: str='dftb') -> QUEDConformer:
        """ 
        Scans conformers one at a time and returns the
        one with the lowest total energy given certain modal.
        """

        keys = self._get_conformer_keys(idx)
        energies = [self._conformer_energy(idx, k, modal) for k in keys]
        return self.get_conformer(idx, int(np.argmin(energies)))