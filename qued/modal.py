# requirements
from abc import ABC, abstractmethod
from typing import Dict

import os
import io
import re
import sys
import time
import json
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from pyscf import gto, lib
from skala.pyscf import SkalaKS
from ase.io import read, write
from ase.units import Hartree
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator, mace_polar

XTB = shutil.which("xtb") or str(Path(sys.executable).with_name("xtb"))
AU2DEBYE = 2.541746

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s\n',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def resolve_device(device: str | None = None) -> str:
    """'cuda' if a GPU is visible (and not overridden), else 'cpu'."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def frontier_orbitals(energies, occupations, n=4):
    """Return (HOMO-n+1 ... HOMO), (LUMO ... LUMO+n-1); NaN-padded if missing."""
    e = np.asarray(energies, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    # Identify the HOMO–LUMO gap
    # HOMO = last occupied orbital (occupation ≠ 0)
    # LUMO = first unoccupied orbital (occupation = 0)
    ihomo = int(np.nonzero(occ > 1e-6)[0].max())
    homos = np.full(n, np.nan)
    lumos = np.full(n, np.nan)
    occ_part = e[max(0, ihomo - n + 1): ihomo + 1]
    homos[n - len(occ_part):] = occ_part
    virt_part = e[ihomo + 1: ihomo + 1 + n]
    lumos[:len(virt_part)] = virt_part
    return homos, lumos

COMPRESS = {"compression": "gzip", "compression_opts": 4}

def _write_group(group: h5py.Group, properties: dict) -> None:
    """Recursively write a (possibly nested) dict into an HDF5 group.
    dict -> subgroup; arrays/lists -> dataset; scalars -> scalar dataset; None -> skipped."""
    for name, data in properties.items():
        name = str(name)
        if name in group:                      # overwrite on re-run
            del group[name]

        if isinstance(data, dict):             # e.g. 'gas' / 'sol' for xTB
            _write_group(group.create_group(name), data)
            continue
        if data is None:                       # h5py cannot store None
            logger.debug(f"{group.name}/{name}: None, not stored")
            continue
        if isinstance(data, str):
            group.create_dataset(name, data=data, dtype=h5py.string_dtype())
            continue

        arr = np.asarray(data)                 # lists, nested lists, numpy scalars/arrays
        if arr.dtype == object:
            raise TypeError(f"{group.name}/{name}: cannot store ragged/mixed data "
                            f"of type {type(data).__name__}")
        if arr.ndim == 0:                      # scalars: compression not allowed
            group.create_dataset(name, data=arr)
        else:
            group.create_dataset(name, data=arr, **COMPRESS)

class QUEDModalCalculator(ABC):
    name: str = "" # modality label, e.g., "dftb", "skala"

    def __init__(self, output_directory='.'):
        if not self.name:
            raise TypeError(f"{type(self).__name__} must define a class attribute `name`")
        
        self.output_path = Path(output_directory).resolve()

    @abstractmethod
    def calculate(self, xyz_block: str, charge: int, **kwargs) -> Path:
        """Run the method and return the path of the raw output file."""
 
    @abstractmethod
    def is_valid(self, raw_output: Path) -> bool:
        """True only if the output is COMPLETE (an empty or truncated file
        must return False; check a method-specific end-of-run marker)."""
 
    @abstractmethod
    def extract(self, raw_output: Path, **kwargs) -> Dict[str, object]:
        """Parse a valid output into {property label: value}, in documented units."""

    def validate_and_cleanup(self, raw_output: Path) -> bool:
        """Validate output and remove it if invalid."""
        if not raw_output.exists():
            return False

        if not self.is_valid(raw_output):
            logger.warning(f"Invalid output: {raw_output}")

            raw_output.unlink(missing_ok=True)
            return False

        return True

    def save_in_hdf5(self, hf_path, properties, molid, confid):
        """Store properties under <molid>/<confid>/<modal name>/...
        Nested dicts (e.g. xTB {'gas': {...}, 'sol': {...}}) become subgroups."""
        with h5py.File(hf_path, "a") as hf:
            conf_group = hf.require_group(f"{molid}/{confid}")
            if self.name in conf_group:        # replace this modal completely
                del conf_group[self.name]
            _write_group(conf_group.create_group(self.name), properties)

class SkalaModalCalculator(QUEDModalCalculator):
    def __init__(self, output_directory='.', device=None):
        self.name = 'skala'
        super().__init__(output_directory=output_directory)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        # Skala compares devices strictly: 'cuda' != 'cuda:0'
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device

    def calculate(self, xyz_block: str, charge: int, **kwargs) -> Path:
        label = kwargs.get('label', 'current_skala')
        output_directory = Path(self.output_path).resolve()
        n_threads = kwargs.get('n_threads', os.cpu_count())
        # Set number of CPU threads
        lib.num_threads(n_threads)
        
        # Directory to store <label>.out file (relevant output)
        skala_directory = output_directory / "skala"
        skala_directory.mkdir(parents=True, exist_ok=True)
                
        start = time.perf_counter()
        raw_output = skala_directory / f'{label}.out'
        # Define Molecule object
        params = dict(basis="def2-tzvp",
                      output=raw_output, verbose=3)
        if charge!=0: 
            params.update(dict(charge=charge))
        if 'I' in xyz_block:
            params.update(dict(ecp={'I':'def2-TZVP'}))
        mol = gto.M(atom=xyz_block, **params)
        mol.unit = 'Angstrom'
        # Define Calculator with Skala
        ks = SkalaKS(mol, xc="skala-1.1", ks_config={"conv_tol": 1e-6},
                     device=self.device)
        # Calculate QM properties: Energies, Dipole, eMOs, and atomic charges
        try:
            # Run calculation
            ks.kernel()
            ks.dump_scf_summary()
            (_, mulliken), dipole = ks.analyze() 
        except Exception as e:
            logger.exception(f"Skala calculation failed")
            raise RuntimeError(
                f"Skala calculation failed: {e}"
            ) from e
        
        elapsed = time.perf_counter() - start
        logger.info(
            f"Molecule {label}: Success. "
            f"Time: {elapsed:.1f} s"
            )
        return raw_output

    def is_valid(self, raw_output: Path) -> bool:
        # Empty file
        if os.path.getsize(raw_output)==0:
            return False
        # Not converged
        with open(raw_output, 'r') as f:
            found = any("converged SCF energy" in line for line in f)
            if not found:
                return False
        return True

    def extract(self, raw_output: Path, **kwargs) -> Dict[str, object]:
        # Read output file
        with open(raw_output, 'r') as f:
            lines = f.readlines()
        properties = dict()

        for i, line in enumerate(lines):
            # Extract energy contributions (in H)
            if line.strip().startswith("Total Energy"):
                properties['total_energy_Eh'] = float(line.split()[4])
            if line.strip().startswith("Nuclear Repulsion Energy"):
                properties['nuclear_repulsion_energy_Eh'] = float(line.split()[4])
            if line.strip().startswith("One-electron Energy"):
                properties['one_electron_energy_Eh'] = float(line.split()[3])
            if line.strip().startswith("Two-electron Coulomb Energy"):
                properties['two_electron_energy_Eh'] = float(line.split()[4])
            if line.strip().startswith("DFT Exchange-Correlation Energy"):
                properties['xc_energy_Eh'] = float(line.split()[4])
            if line.strip().startswith("Empirical Dispersion Energy"):
                properties['empirical_dispersion_energy_Eh'] = float(line.split()[4])
                start = i
                break

        # Extract Molecular Orbital energies. 
        # Energies will be converted from H to eV.
        # Return (HOMO-n+1 ... HOMO), (LUMO ... LUMO+n-1); NaN-padded if missing
        iorbs = kwargs.get('iorbs', 4)
        energy_occupation = []
        for i, line in enumerate(lines[start:]):
            if line.startswith('MO'):
                tokens = [x for x in line.split() if len(x)>0]
                energy_occupation.append([float(tokens[3]), float(tokens[5])])
            if line.startswith(" ** Mulliken atomic charges  **"):
                start += i
                break

        data = np.asarray(energy_occupation)
        energies, occupations = data.T
        homos, lumos = frontier_orbitals(energies, occupations, iorbs)
        properties['homos_eV'] = homos * Hartree
        properties['lumos_eV'] = lumos * Hartree          

        # Extract Mulliken atomic charges
        atom_chg = []
        for line in lines[start:]:
            if line.startswith("charge of"):
                tokens = [x for x in line.split() if len(x)>0]
                atom_chg.append(float(tokens[4]))

            if line.startswith("Dipole moment"):
                tokens = [x for x in line.split() if len(x)>0]
                properties['dipole_vector_debye'] = np.array([float(tokens[-3][:-1]), float(tokens[-2][:-1]), float(tokens[-1])])
        properties['partial_charges'] = np.array(atom_chg)
        return properties

class VibraModalCalculator(QUEDModalCalculator):
    def __init__(self, output_directory='.', model='omol', model_path=None, device=None):
        self.name = 'vibra'
        super().__init__(output_directory=output_directory)
        self.KEEP = ["optimized_geometry.xyz", "vibrational_data.txt"]
        self.model = model
        self.device = resolve_device(device)
        kw = dict(device=self.device, default_dtype="float64")
        if model == 'omol':
            if model_path is None:
                raise ValueError("model='omol' needs model_path (local .model file)")
            self.mace_calculator = MACECalculator(model_paths=model_path, **kw)
        elif model == 'polar-m':
            self.mace_calculator = mace_polar(model="polar-1-m", **kw)
        elif model == 'polar-l':
            self.mace_calculator = mace_polar(model="polar-1-l", **kw)
    
    def calculate(self, xyz_block: str, charge: int, **kwargs) -> Path:
        label = kwargs.get('label', 'current_vibra')
        output_directory = Path(self.output_path).resolve()
        
        # Directory to store <label>.out file (relevant output)
        vibra_directory = output_directory / "vibra"
        vibra_directory.mkdir(parents=True, exist_ok=True)

        # Create independent working directory
        work_directory = output_directory / "work_vibra" / label
        work_directory.mkdir(parents=True, exist_ok=True)

        # Generate Atoms object that will work as input
        # Wrap the string in StringIO to mimic a file
        n_atoms = len(xyz_block.splitlines())
        xyz_file_like = io.StringIO(f'{n_atoms}\n\n{xyz_block}')
        amol = read(xyz_file_like, format="xyz")
        amol.calc = self.mace_calculator
        amol.calc.directory = str(work_directory)
        amol.calc.label = label
        log_file = work_directory / "optimization.log"

        start = time.perf_counter()
        try:
            # Geometry optimization
            # Convergence threshold: Maximum atomic force=0.001 eV/A, Maximum number of opt steps=4000
            qn = BFGS(amol, trajectory=str(work_directory / 'mol.traj'), logfile=str(log_file))
            qn.run(fmax=0.001, steps=2000)
        except Exception as e:
            logger.exception(f"MACE calculation failed in {work_directory}")
            raise RuntimeError(
                f"MACE calculation failed in {work_directory}: {e}"
            ) from e
        # Save optimized geometry
        opt_destination = work_directory / "optimized_geometry.xyz"
        write(opt_destination, amol)

        # Calculate vibrational frequencies (3N-6 nonlinear, 3N-5 linear mol)
        vib = Vibrations(amol, delta=0.01, nfree=4, name=str(work_directory / "vibrations"))
        vib.run()
        # vib.summary()
        # Vibrational energies (complex values in eV)
        vb_ev = vib.get_energies()
        # Vibrational frequencies (potentially complex in cm-1)
        vb_cm = vib.get_frequencies()
        # Extract total energy
        total_energy = amol.get_total_energy()

        # Save vibrational information
        modes_destination = work_directory / "vibrational_data.txt"
        with open(modes_destination, 'a') as o1:
            o1.write("{:>24}".format(label) + "{:>24}".format(total_energy) + "\n")
            o1.write("{:>24}".format("energies-real") + "{:>24}".format("energies-imag") + "{:>24}".format("freq-real") + "{:>24}".format("freq-imag") + "\n")
            for i in range(0, len(vb_ev)):
                o1.write("{:>24}".format(vb_ev.real[i]) + "{:>24}".format(vb_ev.imag[i]) +\
                         "{:>24}".format(vb_cm.real[i]) + "{:>24}".format(vb_cm.imag[i]) + "\n")
        # Generate a Jmol file (vibrational modes that can be visualized using Jmol)
        # vib.write_jmol()

        # Store files in a single zip file
        destination = vibra_directory / f"{label}.zip"
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in self.KEEP:
                f = work_directory / fname
                if f.exists():
                    z.write(f, fname)
        shutil.rmtree(work_directory)

        elapsed = time.perf_counter() - start
        logger.info(
            f"Molecule {label}: Success. "
            f"Time: {elapsed:.1f} s"
            )
        return destination

    def is_valid(self, raw_output: Path) -> bool:
        # Empty file
        if os.path.getsize(raw_output)==0:
            return False
        return True

    def extract(self, raw_output: Path, **kwargs) -> Dict[str, object]:
        properties = dict()

        with zipfile.ZipFile(raw_output) as z:
            # Extract optimized geometry
            atoms = read(io.StringIO(z.read('optimized_geometry.xyz').decode()), 
                                       format='extxyz', index=0)
            properties['opt_atomic_numbers'] = atoms.get_atomic_numbers()
            properties['opt_coordinates'] = atoms.get_positions()
            properties['opt_total_energy'] = atoms.get_potential_energy()

            # Extract the real component of the frequencies
            vibra_file = io.StringIO(z.read('vibrational_data.txt').decode())
            frequencies = np.loadtxt(vibra_file, skiprows=2, usecols=(2,3))
            properties['frequencies_real_cm-1'] = frequencies[:, 0]
            properties['frequencies_imag_cm-1'] = frequencies[:, 1]
    
        return properties

class XTBModalCalculator(QUEDModalCalculator):
    def __init__(self, output_directory='.'):
        self.name = 'xtb'
        super().__init__(output_directory=output_directory)
        self.phases = {"gas": [], "solvent": ["--gbsa", "water"]}
        self.KEEP = ("xtb.log", "xtbout.json", "wbo", "vibspectrum")
        self.LOG_PATTERNS = {
            "total_energy_Eh": r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh",
            "gradient_norm_Eh_bohr": r"\|\s*GRADIENT NORM\s+(-?\d+\.\d+)",
            "homo_lumo_gap_eV": r"\|\s*HOMO-LUMO GAP\s+(-?\d+\.\d+)",
            "scc_energy_Eh": r"::\s*SCC energy\s+(-?\d+\.\d+)",
            "Gsolv_Eh": r"::\s*-> Gsolv\s+(-?\d+\.\d+)",
            "Gelec_Eh": r"::\s*-> Gelec\s+(-?\d+\.\d+)",
            "Gsasa_Eh": r"::\s*-> Gsasa\s+(-?\d+\.\d+)",
            "Ghb_Eh": r"::\s*-> Ghb\s+(-?\d+\.\d+)",
            "Gshift_Eh": r"::\s*-> Gshift\s+(-?\d+\.\d+)",
            "homo_eV": r"(-?\d+\.\d+)\s+\(HOMO\)",
            "lumo_eV": r"(-?\d+\.\d+)\s+\(LUMO\)",
            "fermi_level_eV": r"Fermi-level\s+-?\d+\.\d+\s+Eh\s+(-?\d+\.\d+)\s+eV",
            "C6_au": r"Mol\. C6AA /au.*?:\s*(-?\d+\.\d+)",
            "C8_au": r"Mol\. C8AA /au.*?:\s*(-?\d+\.\d+)",
            "polarizability_au": r"Mol\. α\(0\) /au\s*:\s*(-?\d+\.\d+)",
            "dipole_debye": r"molecular dipole:.*?full:\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)",
            # thermochemistry (only with --hess/--ohess)
            "total_enthalpy_Eh": r"\|\s*TOTAL ENTHALPY\s+(-?\d+\.\d+)",
            "total_free_energy_Eh": r"\|\s*TOTAL FREE ENERGY\s+(-?\d+\.\d+)",
            "zpe_Eh": r"::\s*zero point energy\s+(-?\d+\.\d+)",
            "G_rrho_contrib_Eh": r"::\s*G\(RRHO\) contrib\.\s+(-?\d+\.\d+)",
        }

    def calculate(self, xyz_block: str, charge: int, **kwargs) -> Path:
        label = kwargs.get('label', 'current_xtb')
        output_directory = Path(self.output_path).resolve()
        n_threads = kwargs.get('n_threads', os.cpu_count())

        # Directory to store output files (relevant output)
        xtb_directory = output_directory / "xtb"
        xtb_directory.mkdir(parents=True, exist_ok=True)

        # Create independent working directory
        work_directory = output_directory / "work_xtb_modal" / label
        work_directory.mkdir(parents=True, exist_ok=True)

        # Prepare input xyz file
        lines = [l for l in xyz_block.splitlines() if l.strip()]
        new_xyz_block = f"{len(lines)}\n\n" + "\n".join(lines) + "\n"

        # Run solvent and gas calculations
        for phase, config in self.phases.items():
            start = time.perf_counter()
            d = work_directory / phase
            d.mkdir(parents=True, exist_ok=True)
            (d / "initial.xyz").write_text(new_xyz_block)
            # Run xtb
            log_file = d / "xtb.log"
            command = [
                XTB,
                str(d / "initial.xyz"), '--charge', str(charge), 
                '--json', *config,
                "-P", str(n_threads)
            ]
            with open(log_file, "w") as log:
                proc = subprocess.run(
                        command,
                        cwd=d,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True
                    )
            text = (d / "xtb.log").read_text()
            elapsed = time.perf_counter() - start
            if proc.returncode != 0 or "normal termination" not in text:
                logger.info(
                    f"Molecule {label}: Failed in {d}. "
                    f"Time: {elapsed:.1f} s. See {log_file}"
                )
                return None

        # Store files in a single zip file
        destination = xtb_directory / f"{label}.zip"
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
            for phase in self.phases:
                for fname in self.KEEP:
                    f = work_directory / phase / fname
                    if f.exists():
                        z.write(f, f"{phase}/{fname}")
        shutil.rmtree(work_directory)

        elapsed = time.perf_counter() - start
        logger.info(
            f"Molecule {label}: Success. "
            f"Time: {elapsed:.1f} s"
            )
        
        return destination

    def is_valid(self, raw_output: Path) -> bool:
        # Empty file
        if not zipfile.is_zipfile(raw_output):
            return False
        with zipfile.ZipFile(raw_output) as z:
            return all(
                f"{p}/xtb.out" in z.namelist()
                and b"normal termination of xtb" in z.read(f"{p}/xtb.out")
                for p in self.phases
            )

    def extract(self, raw_output: Path, **kwargs) -> Dict[str, object]:
        # Return (HOMO-n+1 ... HOMO), (LUMO ... LUMO+n-1); NaN-padded if missing
        iorbs = kwargs.get('iorbs', 4)

        properties = dict()
        with zipfile.ZipFile(raw_output) as z:
            for phase in self.phases:
                properties[phase] = dict()
                # Read xtb.log
                text = z.read(f"{phase}/xtb.log").decode("utf-8", errors="replace")
                for key, pat in self.LOG_PATTERNS.items():
                    m = re.findall(pat, text, flags=re.S if key == "dipole_debye" else 0)
                    if m:
                        properties[phase][key] = float(m[-1])
                m = re.findall(r"molecular dipole:.*?full:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", text, flags=re.S)
                if m:
                    dipole_vector_au = [float(v) for v in m[-1][:3]]
                    properties[phase]["dipole_vector_debye"] = [v * AU2DEBYE for v in dipole_vector_au]
                # Read json
                data = json.loads(z.read(f"{phase}/xtbout.json"))
                properties[phase]["partial_charges"] = data.get("partial charges")
                ## Frontier orbitals
                e = np.asarray(data.get("orbital energies / eV"), dtype=float)
                occ = np.asarray(data.get("fractional occupation"), dtype=float)
                ihomo = int(np.nonzero(occ > 1e-6)[0].max())
                homos = np.full(iorbs, np.nan)
                lumos = np.full(iorbs, np.nan)
                occ_part = e[max(0, ihomo - iorbs + 1): ihomo + 1]
                homos[iorbs - len(occ_part):] = occ_part
                virt_part = e[ihomo + 1: ihomo + 1 + iorbs]
                lumos[:len(virt_part)] = virt_part
                properties[phase]["homos_eV"] = homos
                properties[phase]["lumos_eV"] = lumos
                # Other properties
                properties[phase]["electronic_energy_Eh"] = data.get("electronic energy")
                if "vibrational frequencies / rcm" in data:
                    freqs = data["vibrational frequencies / rcm"]
                    properties[phase]["frequencies_cm-1"] = [f for f in freqs if abs(f) > 1e-2]
                    properties[phase]["n_imaginary"] = sum(f < -1e-2 for f in freqs)
                # Read wbo: Wiberg bond orders: "i j order"
                wbo = z.read(f"{phase}/wbo").decode("utf-8", errors="replace")
                properties[phase]["wiberg_bond_orders"] = [
                            [int(i), int(j), float(o)] for i, j, o in (l.split() for l in wbo.split("\n") if l.strip())
                        ]

        return properties