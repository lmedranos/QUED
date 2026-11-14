"""DFTB modal for QUED: DFTB3-3ob + MBD properties for the geometries in an HDF5 file.

Runs in the separate pixi environment `dftb` (DFTB+ 24.1 cannot share an
environment with xtb 6.7.1), so it is self-contained and does not import qued.

Same conventions as qued.modal / generate_datasets.py:
  geometries   <molid>/<confid>/geometry/{atomic_numbers, atomic_coordinates}
  raw output   <output_dir>/dftb/<molid>-<confid>.out   (reused if valid)
  properties   <molid>/<confid>/dftb/...                 (same HDF5 file)

Usage
-----
pixi run -e dftb python scripts/dftb_modal.py -hdf5 out/optimized.h5 -o out \\
    -i db.csv --skf_dir /path/to/3ob-3-1 --n_threads 4

Property units: energies in Hartree (FermiEne, BandEne, h0Ene, sccEne, 3rdEne,
repEne, mbdEne), orbital energies TBeig in eV, dipole TBdip in Debye,
Mulliken charges TBchg in e, NumElec = number of electrons.
"""
import os
import re
import csv
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from ase import Atoms
from ase.calculators.dftb import Dftb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s\n', datefmt='%H:%M:%S')

AU2DEBYE = 2.541746
COMPRESS = {"compression": "gzip", "compression_opts": 4}

# 3ob-3-1 Hubbard derivatives (DFTB3), from dftb.org; previously in the patched ase dftb.py
HUBBARD_DERIVS = {
    'H': -0.1857, 'C': -0.1492, 'N': -0.1535, 'O': -0.1575, 'F': -0.1623,
    'Na': -0.0454, 'Mg': -0.02, 'P': -0.14, 'S': -0.11, 'Cl': -0.0697,
    'K': -0.0339, 'Ca': -0.0340, 'Zn': -0.03, 'Br': -0.0573, 'I': -0.0433,
}


# ------------------------------------------------------------------ HDF5 helpers
def read_geometries(h5_path, molids=None):
    """[(molid, confid, Z, R)] from <molid>/<confid>/geometry; file closed afterwards."""
    out = []
    with h5py.File(h5_path, "r") as hf:
        for molid in (molids if molids is not None else hf.keys()):
            molid = str(molid)
            if molid not in hf:
                logger.warning(f"Molecule {molid}: no geometries in {h5_path}, skipped.")
                continue
            for confid in hf[molid].keys():
                g = hf[molid][confid]["geometry"]
                out.append((molid, confid, g["atomic_numbers"][()], g["atomic_coordinates"][()]))
    return out


def read_charges(csv_path, h5_path, id_label="molID"):
    """molid -> charge: CSV 'charge' column, else HDF5 molecule attribute 'charge', else 0."""
    charges = {}
    if csv_path:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if "charge" in row and row["charge"] not in ("", None):
                    charges[str(row[id_label])] = int(float(row["charge"]))
    with h5py.File(h5_path, "r") as hf:
        for molid in hf.keys():
            if molid not in charges and "charge" in hf[molid].attrs:
                charges[molid] = int(hf[molid].attrs["charge"])
    return charges


def _write_group(group, properties):
    """Recursively write a dict into an HDF5 group (same as qued.modal)."""
    for name, data in properties.items():
        name = str(name)
        if name in group:
            del group[name]
        if isinstance(data, dict):
            _write_group(group.create_group(name), data)
            continue
        if data is None:
            continue
        arr = np.asarray(data)
        if arr.dtype == object:
            raise TypeError(f"{group.name}/{name}: cannot store ragged/mixed data")
        group.create_dataset(name, data=arr, **({} if arr.ndim == 0 else COMPRESS))


# ------------------------------------------------------------------ calculator
class DFTBModalCalculator:
    name = "dftb"
    extension = ".out"
    allowed_species = set(HUBBARD_DERIVS)

    def __init__(self, output_directory=".", skf_dir=None, dftb_command="dftb+"):
        self.output_path = Path(output_directory).resolve()
        skf_dir = skf_dir or os.environ.get("DFTB_PREFIX")
        if not skf_dir or not Path(skf_dir).is_dir():
            raise SystemExit("Slater-Koster files not found: pass --skf_dir or set DFTB_PREFIX")
        self.skf_dir = str(Path(skf_dir).resolve()) + "/"
        self.dftb_command = shutil.which(dftb_command) or dftb_command

    def output_file(self, label: str) -> Path:
        return self.output_path / self.name / f"{label}{self.extension}"

    # Same settings as the previous DFTB modal: DFTB3 (3ob) + H-damping + MBD
    def _parameters(self, symbols, charge):
        params = dict(
            Hamiltonian_SCC='Yes',
            Hamiltonian_MaxSCCIterations=2000,
            Hamiltonian_ThirdOrderFull='Yes',
            Hamiltonian_HCorrection_='Damping',
            Hamiltonian_HCorrection_Exponent=4.05,
            Hamiltonian_Dispersion_='MBD',
            Hamiltonian_Dispersion_Beta=0.83,
            Hamiltonian_Dispersion_KGrid='1 1 1',
            Hamiltonian_Dispersion_NOmegaGrid=25,
            Hamiltonian_Dispersion_ReferenceSet='ts',
            Hamiltonian_HubbardDerivs_='',
            Parallel_='',
            Parallel_UseOmpThreads='Yes',
            Options_='',
            ParserOptions_='',
            ParserOptions_ParserVersion=9,
        )
        for s in sorted(set(symbols)):
            params[f'Hamiltonian_HubbardDerivs_{s}'] = HUBBARD_DERIVS[s]
        if charge != 0:
            params['Hamiltonian_Charge'] = float(charge)
        return params

    def calculate(self, Z, R, charge: int, label: str, n_threads: int = 1) -> Path:
        atoms = Atoms(numbers=Z, positions=R)
        symbols = atoms.get_chemical_symbols()
        unknown = set(symbols) - self.allowed_species
        if unknown:
            raise ValueError(f"No 3ob Hubbard derivative for {sorted(unknown)}")

        work = self.output_path / "work_dftb" / label
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        self.output_file(label).parent.mkdir(parents=True, exist_ok=True)

        # ASE only writes dftb_in.hsd + geometry (max. angular momenta read from the .skf files)
        calc = Dftb(label=label, atoms=atoms, directory=str(work), slako_dir=self.skf_dir,
                    **self._parameters(symbols, charge))
        calc.atoms = atoms.copy()          # write_dftb_in() reads self.atoms
        calc.write_input(atoms)

        start = time.perf_counter()
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        with open(work / f"{label}.out", "w") as log:
            subprocess.run([self.dftb_command], cwd=work, stdout=log,
                           stderr=subprocess.STDOUT, env=env, check=False)

        # Merge the relevant outputs into one file (same layout as before)
        raw_output = self.output_file(label)
        with raw_output.open("w") as out:
            for fname in (f"{label}.out", "detailed.out", "band.out"):
                fpath = work / fname
                if fpath.exists():
                    out.write(f"\n\n===== {fname} =====\n")
                    out.write(fpath.read_text())
        shutil.rmtree(work)
        logger.info(f"Molecule {label}: DFTB done in {time.perf_counter() - start:.1f} s")
        return raw_output

    def is_valid(self, raw_output: Path) -> bool:
        if not raw_output.exists() or raw_output.stat().st_size == 0:
            return False
        text = raw_output.read_text(errors="replace")
        return ("SCC converged" in text and "===== detailed.out =====" in text
                and "===== band.out =====" in text)

    def validate_and_cleanup(self, raw_output: Path) -> bool:
        if not raw_output.exists():
            return False
        if not self.is_valid(raw_output):
            logger.warning(f"Invalid output: {raw_output}")
            raw_output.unlink(missing_ok=True)
            return False
        return True

    @staticmethod
    def _sections(text):
        parts = re.split(r"\n===== (\S+) =====\n", text)
        return {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}

    def extract(self, raw_output: Path, n_homos: int = 4, n_lumos: int = 4) -> Dict[str, object]:
        sec = self._sections(raw_output.read_text(errors="replace"))
        detailed, band = sec["detailed.out"], sec["band.out"]
        p = {}

        # Mulliken charges: block after the " Atom           Charge" header
        lines = detailed.splitlines()
        i0 = next(i for i, l in enumerate(lines) if re.match(r"\s*Atom\s+Charge\s*$", l))
        chg = []
        for l in lines[i0 + 1:]:
            tok = l.split()
            if len(tok) == 2:
                chg.append(float(tok[1]))
            elif chg:
                break
        p['partial_charges'] = np.array(chg)

        def first(pattern, cast=float):
            m = re.search(pattern, detailed, flags=re.M)
            return cast(m.group(1)) if m else None

        num = r"(-?\d+\.\d+(?:[EeDd][+-]?\d+)?)"
        p['Fermi_energy_Eh'] = first(rf"^Fermi level:\s+{num}\s+H")
        p['band_energy_Eh'] = first(rf"^Band energy:\s+{num}\s+H")
        p['number_electrons'] = first(rf"^Input / Output electrons \(q\):\s+\S+\s+{num}")
        p['h0_energy_Eh'] = first(rf"^Energy H0:\s+{num}\s+H")
        p['scc_energy_Eh'] = first(rf"^Energy SCC:\s+{num}\s+H")
        p['3rd_corr_energy_Eh'] = first(rf"^Energy 3rd:\s+{num}\s+H")
        p['repulsive_energy_Eh'] = first(rf"^Repulsive energy:\s+{num}\s+H")
        p['dispersion_energy_Eh'] = first(rf"^Dispersion energy:\s+{num}\s+H")
        p['total_energy_Eh'] = first(rf"^Total energy:\s+{num}\s+H")
        m = re.search(rf"^Dipole moment:\s+{num}\s+{num}\s+{num}\s+au", detailed, flags=re.M)
        p['dipole_vector_debye'] = np.array([float(x) for x in m.groups()]) * AU2DEBYE if m else None


        # Orbital energies (eV) and occupations from band.out (first KPT/spin block)
        eo = []
        for l in band.splitlines():
            tok = l.split()
            if "KPT" in l:
                if eo:
                    break
                continue
            if len(tok) == 3:
                eo.append((float(tok[1]), float(tok[2])))
        e, occ = np.array(eo).T
        lumo = int(np.nonzero(occ < 1e-8)[0][0])
        homos = np.full(n_homos, np.nan)
        lumos = np.full(n_lumos, np.nan)
        occ_part = e[max(0, lumo - n_homos):lumo]
        homos[n_homos - len(occ_part):] = occ_part
        virt_part = e[lumo:lumo + n_lumos]
        lumos[:len(virt_part)] = virt_part
        p['homos_eV'] = homos
        p['lumos_eV'] = lumos
        return p

    def save_in_hdf5(self, properties, molid, confid, hf_path):
        with h5py.File(hf_path, "a") as hf:
            conf_group = hf.require_group(f"{molid}/{confid}")
            if self.name in conf_group:
                del conf_group[self.name]
            _write_group(conf_group.create_group(self.name), properties)


# ------------------------------------------------------------------ main
def available_cpus() -> int:
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-hdf5', '--hdf5_path', required=True, help='HDF5 with geometries; results go here too')
    ap.add_argument('-o', '--output_dir', required=True, help='raw outputs in <output_dir>/dftb/')
    ap.add_argument('-i', '--database_csv_path', help='CSV with molID (and charge) columns')
    ap.add_argument('--id_label', default='molID')
    ap.add_argument('--skf_dir', default=os.environ.get('DFTB_PREFIX'), help='3ob-3-1 folder')
    ap.add_argument('--dftb_command', default='dftb+')
    ap.add_argument('--n_threads', type=int, default=available_cpus())
    ap.add_argument('--n_homos', type=int, default=4)
    ap.add_argument('--n_lumos', type=int, default=4)
    ap.add_argument('--omit_calculation', action='store_true', help='only store existing outputs')
    a = ap.parse_args()

    calc = DFTBModalCalculator(a.output_dir, skf_dir=a.skf_dir, dftb_command=a.dftb_command)
    charges = read_charges(a.database_csv_path, a.hdf5_path, a.id_label)
    molids = None
    if a.database_csv_path:
        with open(a.database_csv_path, newline="") as f:
            molids = [str(r[a.id_label]) for r in csv.DictReader(f)]

    n_done = n_reused = n_failed = 0
    for molid, confid, Z, R in read_geometries(a.hdf5_path, molids):
        label = f"{molid}-{confid}"
        raw = calc.output_file(label)
        if calc.validate_and_cleanup(raw):
            logger.info(f"{label}: dftb output found and valid, skipping calculation")
            n_reused += 1
        elif a.omit_calculation:
            continue
        else:
            try:
                raw = calc.calculate(Z, R, charges.get(molid, 0), label, a.n_threads)
            except Exception:
                logger.exception(f"{label}: dftb calculation failed")
                n_failed += 1
                continue
            if not calc.validate_and_cleanup(raw):
                logger.warning(f"{label}: dftb did not converge / no valid output")
                n_failed += 1
                continue
            n_done += 1
        try:
            props = calc.extract(raw, n_homos=a.n_homos, n_lumos=a.n_lumos)
            calc.save_in_hdf5(props, molid, confid, a.hdf5_path)
        except Exception:
            logger.exception(f"{label}: dftb properties could not be stored")
            n_failed += 1

    logger.info(f"dftb: {n_done} calculated, {n_reused} reused, {n_failed} failed")


if __name__ == "__main__":
    main()
