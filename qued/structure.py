# requirements
import re
import time
import shlex
import subprocess
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.data import atomic_numbers as _SYMBOL_TO_Z

_ENERGY_RE = re.compile(r"^\s*(?:energy:\s*)?([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)(?=\s|$)")
XTB = shutil.which("xtb") or str(Path(sys.executable).with_name("xtb"))
CREST = shutil.which("crest") or str(Path(sys.executable).with_name("crest"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s\n',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class QUEDStructureCalculator:
    @staticmethod
    def smile2xyz(smiles, 
                  filename):
        """
        SMILES -> initial 3D geometry (XYZ file), meant to seed a conformer search
        (e.g. CREST), so one reasonable conformer is all that is needed.
        
        Parameters
        ----------
        - smiles    : SMILES string
        - filename  : output XYZ file path
        
        Returns
        ----------
        True if a 3D geometry was written.
        """
        # Cheap first, more expensive only if the previous strategy failed.
        _EMBED_ATTEMPTS = (
            {},                           # plain ETKDGv3
            {"useRandomCoords": True},    # commonly suggested for large / flexible / ring-rich molecules
        )
        seed, max_iters, timeout= 0xF00D, 2000, 60

        def _embed(mol, seed, timeout):
            # Returns True if a conformer was embedded into mol
            for opts in _EMBED_ATTEMPTS:
                params = AllChem.ETKDGv3()
                params.randomSeed = seed
                if timeout and hasattr(params, "timeout"):   # attribute missing in older RDKit
                    params.timeout = timeout
                for key, value in opts.items():
                    setattr(params, key, value)
                mol.RemoveAllConformers()
                if AllChem.EmbedMolecule(mol, params) == 0:   # -1 means failure; nothing is raised
                    return True
            return False

        def _force_field_cleanup(mol, max_iters):
            # Returns 'MMFF', 'UFF' or None (no parameters), plus whether it converged
            if AllChem.MMFFHasAllMoleculeParams(mol):
                return "MMFF", AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters) == 0
            if AllChem.UFFHasAllMoleculeParams(mol):      # e.g. metals, boron, selenium
                return "UFF", AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters) == 0
            return None, False

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return False
        mol = Chem.AddHs(mol)
    
        if not _embed(mol, seed, timeout):
            logger.warning(f"3D embedding failed for SMILES: {smiles}")
            return False

        method, converged = _force_field_cleanup(mol, max_iters)
        if method is None:
            logger.info(f"No MMFF/UFF parameters, written unoptimised: {smiles}")
        elif not converged:
            logger.info(f"{method} optimisation did not converge in {max_iters} iterations: {smiles}")

        Chem.MolToXYZFile(mol, filename)
        return True

    @staticmethod
    def read_xyz_frames(xyz_path):
        """Yield (coordinates, energy) for each frame of an XYZ file.
        The comment line may be a bare float or 'energy: <float> ...' (xtb)."""
        with open(xyz_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                n_atoms = int(line)
                match = _ENERGY_RE.match(file.readline())
                energy = float(match.group(1)) if match else None
                coordinates = ''.join(file.readline() for _ in range(n_atoms))
                yield coordinates, energy

    @staticmethod
    def get_xyz_frames(xyz_path, indices):
        """Extract selected conformers using zero-based indices"""
        indices = sorted(set(indices))

        for i, frame in enumerate(QUEDStructureCalculator.read_xyz_frames(xyz_path)):
            if i in indices:
                yield i, frame
            if i >= indices[-1]:
                break

    @staticmethod
    def xyzblock_to_array(xyz_block):
        """
        Convert an XYZ block into atomic numbers and coordinates.

        Accepts both the standard format (atom-count line, comment line, atom
        lines) and a bare list of atom lines.
        """
        lines = xyz_block.splitlines()

        # The first non-empty line decides the format
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None:
            raise ValueError("Empty XYZ block")
        head = lines[first].split()

        if len(head) == 1 and head[0].isdigit():
            # Standard header: count line + comment line. Slice by POSITION (the
            # comment line may be empty) and by the stated count (so trailing
            # blank lines are ignored).
            n_atoms = int(head[0])
            atom_lines = lines[first + 2: first + 2 + n_atoms]
            if len(atom_lines) != n_atoms:
                raise ValueError(f"Header says {n_atoms} atoms but only "
                                f"{len(atom_lines)} lines follow the comment line")
        else:
            # No header: every non-empty line is an atom
            atom_lines = [line for line in lines[first:] if line.strip()]
            n_atoms = len(atom_lines)

        atomic_numbers = np.empty(n_atoms, dtype=np.uint8)
        atomic_coords = np.empty((n_atoms, 3), dtype=np.float32)

        for j, line in enumerate(atom_lines):
            tokens = line.split()
            if len(tokens) < 4:
                raise ValueError(f"Malformed atom line: {line!r}")
            if tokens[0] not in _SYMBOL_TO_Z:
                raise ValueError(f"Unknown element symbol {tokens[0]!r} in line: {line!r}")
            atomic_numbers[j] = _SYMBOL_TO_Z[tokens[0]]
            atomic_coords[j] = [float(x) for x in tokens[1:4]]

        return atomic_numbers, atomic_coords

    @staticmethod
    def optimize_geometry(initial_xyz, 
                          output_directory, 
                          xtb_options=None, 
                          n_threads=1):
        """
        Perform geometry optimization with xtb.
        Each molecule is processed in an independent working directory.
        
        Parameters
        ----------
        - initial_xyz: Path to initial XYZ geometry.
        - output_directory: Directory to save generated conformers.
        - xtb_options: Additional xtb command-line arguments.
                         Don't include -T argument. Specify this with n_threads.
        - n_threads: Number of CPU threads allocated to xtb.
        By default, it preserves xtb output files.
        """

        initial_xyz = Path(initial_xyz).resolve()
        output_directory = Path(output_directory).resolve()
        label = initial_xyz.stem

        # Directory to store opt.xyz file (relevant output)
        optimized_directory = output_directory / "optimized"
        optimized_directory.mkdir(parents=True, exist_ok=True)
        
        # Create independent working directory
        work_directory = output_directory / "work_xtb" / label
        work_directory.mkdir(parents=True, exist_ok=True)
        
        start = time.perf_counter()
        # Prepare xtb command
        if xtb_options is None:
            xtb_options = f"--gfn 2 --opt --gbsa water"
        if isinstance(xtb_options, str):
            xtb_options = shlex.split(xtb_options)
        command = [
            XTB,
            str(initial_xyz),
            *xtb_options,
            "-P", str(n_threads)
        ]

        # Run xtb
        log_file = work_directory / "xtb.log"
        try:
            with open(log_file, "w") as log:
                subprocess.run(
                    command,
                    cwd=work_directory,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start
            logger.info(
                f"Molecule {label}: Failed (return code {e.returncode}). "
                f"Time: {elapsed:.1f} s. See {log_file}"
            )
            return None

        # Check that geometry output exists
        optimized_file = work_directory / "xtbopt.xyz"
        # and that the optimization converged
        converged = Path(work_directory, ".xtboptok").exists()
        if not optimized_file.is_file() or not converged:
            logger.info(f"Molecule {label}: Failed. No xtbopt.xyz found.")
            return None

        # Move optimized geometry to final destination
        destination = optimized_directory / f"{label}.xyz"
        shutil.move(str(optimized_file), str(destination))

        # Preserve additional xtb output files
        # Directory to store all output files (can be deleted after inspection)
        xtb_output_directory = output_directory / "out_xtb" / label
        xtb_output_directory.mkdir(parents=True, exist_ok=True)
        for file in work_directory.iterdir():
            if file.is_file():
                shutil.move(str(file), xtb_output_directory / file.name)
        work_directory.rmdir()

        elapsed = time.perf_counter() - start
        
        logger.info(
            f"Molecule {label}: Success. "
            f"Time: {elapsed:.1f} s"
            )

    @staticmethod
    def conformational_search(initial_xyz,
                              output_directory, 
                              crest_options=None, 
                              n_threads=1):
        """
        Perform conformational search with CREST.

        Parameters
        ----------
        - initial_xyz: Path to initial XYZ geometry.
        - output_directory: Directory to save generated conformers.
        - crest_options: Additional CREST command-line arguments.
                         Don't include -T argument. Specify this with n_threads.
        - n_threads: Number of CPU threads allocated to CREST.
        By default, it preserves CREST output files.
        """

        initial_xyz = Path(initial_xyz).resolve()
        output_directory = Path(output_directory).resolve()
        label = initial_xyz.stem

        # Directory to store crest_conformers.xyz file (relevant output)
        conformer_directory = output_directory / "conformers"
        conformer_directory.mkdir(parents=True, exist_ok=True)

        # Create independent working directory
        work_directory = output_directory / "work_crest" / label
        work_directory.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        # Prepare CREST command
        if crest_options is None:
            crest_options = f"-gfn2 -gbsa h2o -mrest 5 -rthr 0.1 -ewin 12.0 -mquick -norotmd"    
        if isinstance(crest_options, str):
            crest_options = shlex.split(crest_options)
        command = [
            CREST,
            str(initial_xyz),
            *crest_options,
            "-T", str(n_threads)
        ]

        # Run CREST
        log_file = work_directory / "crest.log"
        try:
            with open(log_file, "w") as log:
                subprocess.run(
                    command,
                    cwd=work_directory,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start
            logger.info(
                f"Molecule {label}: Failed (return code {e.returncode}). "
                f"Time: {elapsed:.1f} s. See {log_file}"
            )
            return None

        # Check that conformer output exists
        conformer_file = work_directory / "crest_conformers.xyz"
        if not conformer_file.is_file():
            logger.info(f"Molecule {label}: Failed. No crest_conformers.xyz found.")
            return None

        # Get number of conformers from cre_members file
        cre_file = work_directory / "cre_members"
        with open(cre_file, 'r') as f:
            n_conformers = int(f.readline().strip())

        # Move conformers to final destination
        destination = conformer_directory / f"{label}.xyz"
        shutil.move(str(conformer_file), str(destination))

        # Preserve additional CREST output files
        # Directory to store all output files (can be deleted after inspection)
        crest_output_directory = output_directory / "out_crest" / label
        crest_output_directory.mkdir(parents=True, exist_ok=True)
        for file in work_directory.iterdir():
            if file.is_file():
                shutil.move(str(file), crest_output_directory / file.name)
        work_directory.rmdir()

        elapsed = time.perf_counter() - start

        logger.info(
            f"Molecule {label}: Success. "
            f"Conformers: {n_conformers}. "
            f"Time: {elapsed:.1f} s"
            )

class QUEDDatasetStructureCalculator:
    def __init__(self, 
                 csv_path,
                 smiles_label: str = 'smile',
                 id_label: Optional[str] = 'molID',
                 output_directory: str = '.'):
        """
        Parameters
        ----------
        csv_path : str or Path
            Path to CSV file with molecule-level data (smiles, target values)
        smiles_label : str, optional. Default: smile
        id_label : str, optional. Default: molID
            Column names in CSV file to specify SMILE, target property, and molecule index
        """

        # Record database filepath
        self.csv_path = Path(csv_path)
        # Read CSV to get SMILES
        self.df = pd.read_csv(self.csv_path)
        if id_label in self.df.columns:
            self.ids = self.df[id_label].astype(str).tolist()
        else:
            self.ids = [str(i) for i in range(len(self.df))]
        if smiles_label in self.df.columns:
            self.smiles = self.df[smiles_label].tolist()
        else:
            raise KeyError(f"{smiles_label} is missing from {csv_path}")
        self._n = len(self.df)
        self.output_path = Path(output_directory).expanduser().resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)   # optional: create it if missing
        logger.info(f'Output directory: {self.output_path}')

    # Initial 3D coordinates with RDKit of all molecules in the dataset
    def get_initial_coordinates(self):
        self.initial_path = self.output_path / "initial"
        self.initial_path.mkdir(parents=True, exist_ok=True)
        
        # Generate initial XYZ files for all molecules in database
        n_initial_generated = 0
        for molid, smiles in zip(self.ids, self.smiles):
            xyz_path = self.initial_path / f"{molid}.xyz"
            xyz_generated = QUEDStructureCalculator.smile2xyz(smiles, xyz_path)
            if xyz_generated:
                n_initial_generated += 1
        logger.info(f'Initial coordinates generated for: {n_initial_generated}/{len(self.ids)} molecules')

    # Optimize intial geometries 
    def get_optimized_geometry(self, idx, xtb_options=None, n_threads=1):
        initial_xyz = self.initial_path / f"{self.ids[idx]}.xyz"
        charge = self.df.charge[idx] if 'charge' in self.df.columns else 0
        if xtb_options is None:
            xtb_options = f"--charge {charge} --gfn 2 --opt --gbsa water"
        QUEDStructureCalculator.optimize_geometry(initial_xyz,
                                                  self.output_path,
                                                  xtb_options,
                                                  n_threads)

    # Conformational search with CREST on a molecule in the dataset
    def get_conformers(self, idx, crest_options=None, n_threads=1):
        initial_xyz = self.initial_path / f"{self.ids[idx]}.xyz"
        charge = self.df.charge[idx] if 'charge' in self.df.columns else 0
        if crest_options is None:
            crest_options = f"-chrg {charge} -gfn2 -gbsa h2o -mrest 5 -rthr 0.1 -ewin 12.0 -mquick -norotmd"
        QUEDStructureCalculator.conformational_search(initial_xyz,
                                                      self.output_path,
                                                      crest_options,
                                                      n_threads)

    def select_conformers(self, by='stable'):
        conformer_directory = self.output_path / "conformers"
        if not conformer_directory.is_dir():
            logger.info(f"Run conformational_search first to get conformers.")
            return None

        molIDs, confIDs, charges, xTBenergies = [], [], [], []
        # Extract conformer with the lowest xTB energy (the first in the CRE)
        if by=='stable':
            for idx, molid in enumerate(self.ids):
                charge = self.df.charge[idx] if 'charge' in self.df.columns else 0
                conformers_path = conformer_directory / f"{molid}.xyz"
                if not conformers_path.is_file():
                    continue
                _, energy = next(self.read_xyz_frames(conformers_path))
                molIDs.append(molid)
                confIDs.append(0)
                charges.append(charge)
                xTBenergies.append(energy)
        # Other selection methods can be implemented

        # Write CSV file with indices of selected conformers
        selection_path = self.output_path / f"{self.csv_path.stem}_{by}.csv"
        df = pd.DataFrame({'molID': molIDs, 'confID': confIDs, 'charge': charges, 'xTBenergy': xTBenergies})
        df.to_csv(selection_path, index=False)
        return selection_path

    def save_in_hdf5(self, molid, confid, hf_path=None, mode='optimized'):
        # Save geometry in HDF5 file
        if hf_path is None:
            hf_path = self.output_path / f"{mode}.h5"
        compress_mode = {'compression':"gzip", 'compression_opts':4}
        
        with h5py.File(hf_path, 'a') as hf:
            conformers_path = self.output_path / mode / f"{molid}.xyz"
            
            # Hierarchy: molecule > conformer (index according their orden in the CRE)
            molecule_group = hf.require_group(str(molid))
            conf_group = molecule_group.require_group(str(confid))
            
            # Modal: geometry (for geometric descriptors, e.g., MACE)
            g = conf_group.require_group('geometry')
            xyz_block, xtb_energy = next(QUEDStructureCalculator.get_xyz_frames(conformers_path, [confid]))[1]
            Z, xyz = QUEDStructureCalculator.xyzblock_to_array(xyz_block)
            geom_data = {'atomic_numbers': Z, 'atomic_coordinates': xyz}
            for name, data in geom_data.items():
                if name in g:
                    del g[name]
                g.create_dataset(name, data=data, **compress_mode)
            # Store xTB energy
            if 'xTBEne' in g:
                del g['xTBEne']
            g.create_dataset('xTBEne', data=xtb_energy)