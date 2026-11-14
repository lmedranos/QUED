import os
import logging
import argparse
from pathlib import Path

import h5py
import numpy as np
from ase.data import chemical_symbols

from qued.structure import QUEDDatasetStructureCalculator, QUEDStructureCalculator
from qued.modal import XTBModalCalculator, SkalaModalCalculator, VibraModalCalculator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s\n',
    datefmt='%H:%M:%S'
)


def available_cpus() -> int:
    """CPUs actually allocated to this job (Slurm), not all cores of the node."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    try:
        return len(os.sched_getaffinity(0))     # Linux
    except AttributeError:
        return os.cpu_count() or 1              # macOS


def read_xyz_blocks(h5_path, molids=None):
    """Return [(molid, confid, xyz_block)] from <molid>/<confid>/geometry.
    Reads everything and closes the file, so results can be written to the
    same HDF5 afterwards (h5py cannot open one file for reading and writing
    at the same time)."""
    blocks = []
    with h5py.File(h5_path, "r") as hf:
        for molid in (molids if molids is not None else hf.keys()):
            molid = str(molid)
            if molid not in hf:
                logger.warning(f"Molecule {molid}: no geometries in {h5_path}, skipped.")
                continue
            for confid in hf[molid].keys():
                g = hf[molid][confid]["geometry"]
                Z = g["atomic_numbers"][()]
                R = g["atomic_coordinates"][()]          # Angstrom, as read from the xyz files
                xyz_block = "\n".join(
                    f"{chemical_symbols[int(z)]:<2s} {x:16.10f} {y:16.10f} {w:16.10f}"
                    for z, (x, y, w) in zip(Z, R)
                )
                blocks.append((molid, confid, xyz_block))
    return blocks


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--database_csv_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-hdf5', '--output_hdf5_path', type=str, required=True)
    parser.add_argument('--mace_model_path', type=str)
    # Geometries (one source per HDF5 file: optimized and conformer 0 share confid 0)
    geom = parser.add_mutually_exclusive_group()
    geom.add_argument('--calc_opt_geometry', default=False, action='store_true')
    geom.add_argument('--calc_conformers', default=False, action='store_true')
    parser.add_argument('--select_stable_conformer', default=False, action='store_true')
    parser.add_argument('--select_lowest_energy_conformers', default=False, action='store_true')
    parser.add_argument('--select_random_conformers', default=False, action='store_true')
    parser.add_argument('--max_n_conformers', default=30, type=int)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for --select_random_conformers (same seed -> same conformers)')
    # Modal (one per run)
    parser.add_argument('--omit_calculation', default=False, action='store_true',
                        help='only extract/store outputs that already exist')
    modal = parser.add_mutually_exclusive_group()
    modal.add_argument('--calc_skala', default=False, action='store_true')
    modal.add_argument('--calc_xtb', default=False, action='store_true')
    modal.add_argument('--calc_vibra', default=False, action='store_true')
    return parser


# Read parser
arguments = get_parser().parse_args()
n_cpus = available_cpus()

# Input
csv_path = Path(arguments.database_csv_path)
# Output
out_dir = Path(arguments.output_dir)
hdf5_output = Path(arguments.output_hdf5_path)

# Calculate geometries
geom_calculator = QUEDDatasetStructureCalculator(
    csv_path,
    smiles_label='smile',
    id_label='molID',
    output_directory=str(out_dir)
    )
df = geom_calculator.df
molids = geom_calculator.ids                       # str, same order as the CSV
charge_of = {m: int(c) for m, c in zip(molids, df['charge'])} if 'charge' in df \
            else {m: 0 for m in molids}

# Initial geometries (get_initial_coordinates also sets initial_path, needed below)
geom_calculator.initial_path = geom_calculator.output_path / "initial"
if not geom_calculator.initial_path.exists():
    geom_calculator.get_initial_coordinates()

# Optimize geometry
if arguments.calc_opt_geometry:
    for idx, molid in enumerate(molids):
        xyz_path = geom_calculator.output_path / 'optimized' / f"{molid}.xyz"
        if xyz_path.exists():
            logger.info(f"Molecule {molid}: Omit optimization.")
        else:
            geom_calculator.get_optimized_geometry(idx, n_threads=n_cpus)
        geom_calculator.save_in_hdf5(molid, 0, hdf5_output, mode='optimized')

# Generate conformers
if arguments.calc_conformers:
    selections = [arguments.select_stable_conformer,
                  arguments.select_lowest_energy_conformers,
                  arguments.select_random_conformers]
    if sum(selections) != 1:
        raise SystemExit("--calc_conformers needs exactly one --select_* option")

    for idx, molid in enumerate(molids):
        xyz_path = geom_calculator.output_path / 'conformers' / f"{molid}.xyz"
        if xyz_path.exists():
            logger.info(f"Molecule {molid}: Omit conformational search.")
            n_conformers = sum(1 for _ in QUEDStructureCalculator.read_xyz_frames(xyz_path))
        else:
            n_conformers = geom_calculator.get_conformers(idx, n_threads=n_cpus)

        n_sel = min(n_conformers, arguments.max_n_conformers)
        if arguments.select_stable_conformer:
            conf_ids = [0]                                   # lowest energy (CREST order)
        elif arguments.select_lowest_energy_conformers:
            conf_ids = list(range(n_sel))
        else:
            # Reproducible per molecule: re-running picks the same conformers
            rng = np.random.default_rng([arguments.seed, idx])
            conf_ids = sorted(rng.choice(n_conformers, size=n_sel, replace=False).tolist())
        for confid in conf_ids:
            geom_calculator.save_in_hdf5(molid, confid, hdf5_output, mode='conformers')

assert hdf5_output.exists(), "Geometries have to be stored beforehand"

# Choose modal
kwargs = {}
if arguments.calc_xtb:
    modal_calculator = XTBModalCalculator(output_directory=str(out_dir))
    kwargs.update({'iorbs': 8})
elif arguments.calc_skala:
    modal_calculator = SkalaModalCalculator(output_directory=str(out_dir))
    kwargs.update({'iorbs': 8})
elif arguments.calc_vibra:
    if not arguments.mace_model_path:
        raise SystemExit("--calc_vibra needs --mace_model_path")
    modal_calculator = VibraModalCalculator(output_directory=str(out_dir),
                                            model_path=Path(arguments.mace_model_path))
else:
    logger.info("No modal selected (--calc_xtb / --calc_skala / --calc_vibra). Done.")
    raise SystemExit(0)
logger.info(f"Calculating modal: {modal_calculator.name} ({n_cpus} CPU threads)")

# Calculate (or reuse) and store modal properties
n_done = n_reused = n_failed = 0
for molid, confid, xyz_block in read_xyz_blocks(hdf5_output, molids=molids):
    label = f'{molid}-{confid}'
    modal_out = modal_calculator.output_file(label)

    if modal_calculator.validate_and_cleanup(modal_out):
        logger.info(f"{label}: {modal_calculator.name} output found and valid, skipping calculation")
        n_reused += 1
    elif arguments.omit_calculation:
        logger.info(f"{label}: {modal_calculator.name} output missing, not calculated (--omit_calculation)")
        continue
    else:
        try:
            modal_out = modal_calculator.calculate(xyz_block, charge_of[molid],
                                                   label=label, n_threads=n_cpus)
            if modal_out is None or not modal_calculator.validate_and_cleanup(Path(modal_out)):
                logger.warning(f"{label}: {modal_calculator.name} produced no valid output")
                n_failed += 1
                continue
            n_done += 1
        except Exception:
            logger.exception(f"{label}: {modal_calculator.name} calculation failed")
            n_failed += 1
            continue

    try:
        properties = modal_calculator.extract(modal_out, **kwargs)
        modal_calculator.save_in_hdf5(properties, molid, confid, hdf5_output)
    except Exception:
        logger.exception(f"{label}: {modal_calculator.name} could not be stored")
        n_failed += 1

logger.info(f"{modal_calculator.name}: {n_done} calculated, {n_reused} reused, {n_failed} failed")
