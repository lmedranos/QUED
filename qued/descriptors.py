# requirements
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import Descriptors
from dscribe.descriptors import SOAP
from scipy.special import xlogy
from scipy.constants import k
from mace.calculators import MACECalculator, mace_polar
from mace.calculators.mace import o3, extract_invariant

from qued.data import QUEDDataset

class QUEDFeaturizer(ABC):
    def __init__(self):
        """ 
        Defines shared 'compute descriptor logic'
        """
        
        # only known after first call
        self._feature_names: Optional[List[str]] = None
        self._dataset_features_cache: Optional[np.ndarray] = None
        # By default, we work with conformer-level descriptors
        self.fetch_conformer = True 

    @abstractmethod
    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def names(self) -> Optional[Union[List[str], str]]:
        """
        Optional. One of:
          - a list of per-dimension names (validated against output length),
          - a single string tag (expanded to "<tag>_0", "<tag>_1", ...),
          - None (falls back to the class name as the tag).
        """
        return None

    def _expand_names(self, vec: np.ndarray) -> List[str]:
        names = self.names
        if isinstance(names, list):
            if len(names) != len(vec):
                raise ValueError(
                    f"{type(self).__name__}.names has {len(names)} entries "
                    f"but compute() returned {len(vec)} values - check for a "
                    f"property that contributes more/fewer numbers than its "
                    f"declared name(s)."
                )
            return names
        tag = names if isinstance(names, str) else type(self).__name__
        return [f"{tag}_{i}" for i in range(len(vec))]

    def __call__(self, molecule=None, conformer=None) -> np.ndarray:
        vec = np.atleast_1d(self.compute(molecule=molecule, conformer=conformer))
        expanded = self._expand_names(vec)
        if self._feature_names is None:
            self._feature_names = expanded
        return vec

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError(
                f"Call {type(self).__name__} at least once (directly, or via "
                f"featurize_dataset) before requesting feature_names."
            )
        return self._feature_names

class CompositeFeaturizer(QUEDFeaturizer):
    """ 
    Concatenates a list of QUEDFeaturizer blocks 
    (which may themselves be single descriptors OR other 
    CompositeFeaturizers - nesting is fine) into one feature vector.
    """

    def __init__(self, blocks: Sequence[QUEDFeaturizer]):
        super().__init__()
        self.blocks = list(blocks)
        self.fetch_conformer = any(block.fetch_conformer for block in self.blocks)

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        return np.concatenate([block(molecule=molecule, conformer=conformer) for block in self.blocks])

    @property
    def names(self) -> List[str]:
        return [name for block in self.blocks for name in block.feature_names]

def featurize_dataset(featurizer, dataset: QUEDDataset,
                      conf_selector: Union[str, int, Dict[int, int]] = 'lowest_energy'):
    """ 
    Get feature vectors for a dataset.
    ----------
    featurizer : QUEDFeaturizer, CompositeFeaturizer
        Function that construct/calculates representation vector 
    dataset : QUEDDataset
        Dataset over which compute descriptors
    conf_selector: 
        - 'all' for multi-conformer featurizers
        - 'lowest_energy' conformer
        - int: conformer index for every molecule
        - dict {molecule_idx: conformer_idx}
    ----------
    Returns: Matric with one representation per molecule
    """

    rows = []
    for i in range(len(dataset)):
        try:
            molecule = dataset.get_mol(i)
        except:
            molecule = None

        # Determine which conformers to use
        if not featurizer.fetch_conformer or conf_selector == 'all':
            conf_idx = None
        elif conf_selector=='lowest_energy':
            conf_idx = dataset.get_lowest_energy_conformer(i).conformer_index
        elif isinstance(conf_selector, int):
            conf_idx = conf_selector
        elif isinstance(conf_selector, dict):
            conf_idx = conf_selector.get(i, 0)
        else:
            raise ValueError(f"Unknown conf_selector: {conf_selector}")

        # Generate features
        if conf_selector == 'all':
            conformer = dataset.get_all_conformers(i)
        elif conf_idx is None:
            conformer = None
        else:
            conformer = dataset.get_conformer(i, conf_idx)

        rows.append(featurizer(molecule=molecule, conformer=conformer))

    featurizer._dataset_features_cache = np.stack(rows)
    return np.stack(rows)

#### QM DESCRIPTORS ####
QM_PROPS = {'dftb' : {'global_props': ['FermiEne', 'BandEne', 'NumElec'] + \
                            ['h0Ene', 'sccEne', '3rdEne', 'mbdEne'] + \
                            ['TBdip'],
                      'eMO_props': 'TBeig',
                      'atomic_props': 'TBchg',
                      'n_homos': 4, 'n_lumos': 4}, # discard repEne
            'skala': {'global_props': ['el1Ene', 'el2Ene', 'xcEne', 'repEne', 'disEne'] +\
                            ['DFTdip'],
                      'eMO_props': 'DFTeig',
                      'atomic_props': 'DFTchg',
                      'n_homos': 8, 'n_lumos': 8}}

class QMDescriptor(QUEDFeaturizer):
    def __init__(self, modal):
        """ 
        DQM descriptor with DFTB3+MBD- or Skala-calculated properties. 
        modal has to be specified (e.g., 'dftb', 'skala')
        ! repEne should be omitted in DFTB
        """
        
        super().__init__()
        self.modal = modal
        # Properties concatenated as-is, except TBdip which is reduced to
        # its norm (a 3-vector doesn't belong flattened into a "scalar
        # properties" block).
        self.global_props = QM_PROPS[self.modal]['global_props']
        self.eMO_props = QM_PROPS[self.modal]['eMO_props']
        self.atomic_props = QM_PROPS[self.modal]['atomic_props']

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        descriptor = []
        properties = conformer.properties[self.modal]
        # Global properties: concatenate + norm(dipole moment)
        for prop in self.global_props:
            if properties[prop].size>1:
                descriptor.append(np.array([np.linalg.norm(properties[prop])]))
            else:
                descriptor.append(np.atleast_1d(properties[prop]))
        # Electronic properties
        descriptor.append(np.atleast_1d(properties[self.eMO_props]))
        # Atomic properties
        mulliken = properties[self.atomic_props]
        descriptor.append(np.quantile(mulliken, [0, 0.25, 0.5, 0.75, 1]))

        return np.concatenate(descriptor)

    @property
    def names(self) -> List[str]:
        base_names = self.global_props +\
               [f'HOMO-{i}' for i in range(QM_PROPS[self.modal]['n_homos']-1,0,-1)] + ['HOMO'] +\
               ['LUMO'] + [f'LUMO+{i}' for i in range(1, QM_PROPS[self.modal]['n_lumos'])] +\
               ['TBchgmin', 'TBchg25', 'TBchgmed', 'TBchg75', 'TBchgmax']
        return [f"{self.modal}{n}" for n in base_names]

#### RDKIT DESCRIPTORS
RDKIT_PROPS = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'MolMR', 'TPSA', 'MolLogP', 'qed']

class RDkitDescriptor(QUEDFeaturizer):
    """
    Descriptor calculated using RdKit features.
    Topology-based
    """

    def __init__(self):
        super().__init__()
        self.descriptor_labels = RDKIT_PROPS
        self.fetch_conformer = False
        self.modal = 'rdkit'

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        descriptor = []
        if molecule is None:
            descriptor = [np.nan] * len(self.descriptor_labels)
        else:
            all_descriptors = Descriptors.CalcMolDescriptors(molecule)
            descriptor = [all_descriptors[prop] for prop in self.descriptor_labels]
        return np.array(descriptor)

    @property
    def names(self) -> List[str]:
        return self.descriptor_labels

#### GEOMETRIC DESCRIPTORS ####
class MACEDescriptor(QUEDFeaturizer):
    """
    Descriptor generated with MACE embeddings.
    Choose between omol or polar
    """

    def __init__(self, model='omol', invariants_only=True, num_layers=-1, include_electronic=False):
        super().__init__()
        self.modal = 'geometry'
        self.model = model
        self.invariants_only = invariants_only
        self.num_layers = num_layers
        self.include_electronic = include_electronic
        if model=='omol':
            # Rewrite model_path accordingly
            # It can also be loaded with
            # from mace.calculators import mace_omol
            # calc = mace_omol(model="extra_large", device="cpu")
            # but it takes ~1min, this is faster
            model_path = '/Users/ahinostrozac/Documents/programs/confs/MACE-omol-0-extra-large-1024.model'
            # model_path = '/data/horse/ws/alhi416g-confs/MACE-omol-0-extra-large-1024.model'
            self.calc = MACECalculator(model_paths=model_path, device='cpu')
        elif model=='polar-m':
            self.calc = mace_polar(model="polar-1-m", device="cpu", default_dtype="float64")
        elif model=='polar-l':
            self.calc = mace_polar(model="polar-1-l", device="cpu", default_dtype="float64")
        else:
            raise ValueError(f"{model} is not a valid option. Choose omol or polar")

    def _get_embeddings(self, atoms=None):
        # Get number of layers from architecture
        num_interactions = getattr(self.calc.models[0], "num_interactions", None)
        if num_interactions is None:
            num_interactions = len(self.calc.models[0].interactions)
        num_interactions = int(num_interactions)
        if self.num_layers == -1:
            self.num_layers = num_interactions

        # Adapt input data
        batch = self.calc._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.calc.models]

        # Extract invariant features
        irreps_out = o3.Irreps(str(self.calc.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = num_invariant_features  # last layer has no equivariant part
        if self.invariants_only:
            descriptors = [
                extract_invariant(
                    descriptor, num_layers=self.num_layers,
                    num_features=num_invariant_features, l_max=l_max,
                )
            for descriptor in descriptors
            ] 
        to_keep = np.sum(per_layer_features[:self.num_layers])
        descriptors = [descriptor[:, :to_keep].detach().cpu().numpy() for descriptor in descriptors]
        return descriptors[0] if self.calc.num_models == 1 else descriptors

    def _get_electronic_props(self, atoms=None):
        energy = atoms.get_potential_energy()
        mu = np.linalg.norm(self.calc.results["dipole"])
        global_props = np.array([energy, mu])
        # Full multipole array, shape (n_atoms, 4)
        p = self.calc.results["density_coefficients"]
        # atomic monopole charges
        atomic_charges = p[:, 0]
        charges_stats = np.quantile(atomic_charges, [0, 0.25, 0.5, 0.75, 1])
        # cartesian atomic dipoles (px, py, pz)
        atomic_dipoles = p[:, [3, 1, 2]]
        atomic_dipoles = np.linalg.norm(atomic_dipoles, axis=1)
        dipole_stats = np.quantile(atomic_dipoles, [0, 0.25, 0.5, 0.75, 1])
        return np.concat([global_props, charges_stats, dipole_stats])
        
    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        atoms = Atoms(conformer.atomic_numbers, conformer.atomic_coordinates)
        # Get atomic embeddings
        atomic_descriptors = self._get_embeddings(atoms)
        atomic_descriptors = np.mean(atomic_descriptors, axis=0)
        # Get electronic properties (if required)
        if self.include_electronic:
            # Calculate molecular charge
            total_charge = sum(a.GetFormalCharge() for a in molecule.GetAtoms())
            atoms.info["charge"] = total_charge
            atoms.calc = self.calc
            electronic_descriptors = self._get_electronic_props(atoms)
            return np.concat([atomic_descriptors, electronic_descriptors])
        else:
            return atomic_descriptors

    @property
    def names(self) -> str:
        basename = f'MACE{self.model.capitalize()}'
        if self.include_electronic:
            basename += 'Elect'
        return basename

class SOAPDescriptor(QUEDFeaturizer):
    def __init__(self, dataset, periodic=False, r_cut=6.0, n_max=8, l_max=6, sigma=1.0, average="outer"):
        super().__init__()
        self.modal = 'geometry'
        self.configuration = dict(periodic=periodic, r_cut=r_cut,
                                  n_max=n_max, l_max=l_max, 
                                  sigma=sigma, average=average)

        # Get list of species from dataset
        self.species = []
        for i in range(len(dataset)):
            mol = dataset.get_mol(i)
            mol = Chem.AddHs(mol)
            atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
            self.species = list(set(self.species + atom_types))
        self.calculator = SOAP(species=self.species, **self.configuration)

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        assert self.species is not None, "species have to be defined first"

        atoms = Atoms(conformer.atomic_numbers, conformer.atomic_coordinates)
        return self.calculator.create(atoms, n_jobs=-1)

    @property
    def names(self) -> str:
        return 'SOAP'

#### MULTI-CONFORMER DESCRIPTORS ####
class EnsembleFeaturizer(QUEDFeaturizer):
    """
    Multi-Conformer-level descriptors
    """

    def __init__(self):
        super().__init__()
        self.modal = 'ensemble'
        self.energy_key = 'xTBEne'

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        Hartree = 4.359744722e-18 #J
        kBT = k * 300 / Hartree

        # Conformer should be a list of conformers
        energies = np.array([conf.properties[self.modal][self.energy_key] for conf in conformer])
        delta_energies = energies - energies.min()
        populations = np.exp(-delta_energies / kBT)
        populations /= populations.sum()
        entropy = - np.sum(xlogy(populations, populations))
        neff = np.exp(entropy)

        return np.array([populations[np.argmin(energies)], entropy, neff])

    @property
    def names(self) -> List[str]:
        return ['pop_minenergy', 'entropy', 'Neff']

class QMEnsembleDescriptor(QUEDFeaturizer):
    def __init__(self):
        """
        DQM descriptor for the set of conformers. 
        E[X] and Var[X] of the distribution of conformers.
        """
        super().__init__()

    def compute(self, molecule=None, conformer=None) -> np.ndarray:
        descriptor = []
        # Global properties: Fermi energy, band energy (mean)
        for key in ['FermiEne', 'BandEne']:
            prop_values = np.array([conf.properties['dftb'][key] for conf in conformer])
            descriptor.append(np.atleast_1d(prop_values.mean()))
        # Global properties: Number of electrons
        descriptor.append(np.atleast_1d(conformer[0].properties['dftb']['NumElec']))
        # Global properties: Energy components (mean)
        for key in ['h0Ene', 'sccEne', '3rdEne', 'mbdEne']:
            prop_values = np.array([conf.properties['dftb'][key] for conf in conformer])
            descriptor.append(np.atleast_1d(prop_values.mean()))
        # Global properties: Dipole moment (mean and variance)
        prop_values = np.vstack([conf.properties['dftb']['TBdip'] for conf in conformer])
        descriptor.append(np.atleast_1d(np.linalg.norm(prop_values, axis=1).mean()))
        descriptor.append(np.atleast_1d(np.linalg.norm(prop_values, axis=1).var()))
        # Electronic properties: TB Eigenvalues (mean)
        prop_values = np.vstack([conf.properties['dftb']['TBeig'] for conf in conformer])
        descriptor.append(prop_values.mean(axis=0))
        # Atomic properties: TB Mulliken charges (mean)
        mulliken = np.vstack([conf.properties['dftb']['TBchg'] for conf in conformer])
        prop_values = np.quantile(mulliken, [0, 0.25, 0.5, 0.75, 1], axis=1)
        descriptor.append(prop_values.mean(axis=1).T)
        return np.concatenate(descriptor)

    @property
    def names(self) -> List[str]:
        base_names = ['FermiEne_Mean', 'BandEne_Mean', 'NumElec']
        base_names += [x+'_Mean' for x in ['h0Ene', 'sccEne', '3rdEne', 'mbdEne']]
        base_names += ['TBdip_Mean', 'TBdip_Var']
        base_names += [f'HOMO-{i}_Mean' for i in range(QM_PROPS['dftb']['n_homos']-1,0,-1)] + ['HOMO_Mean']
        base_names += ['LUMO_Mean'] + [f'LUMO+{i}_Mean' for i in range(1, QM_PROPS['dftb']['n_lumos'])]
        base_names += [x+'_Mean' for x in ['TBchgmin', 'TBchg25', 'TBchgmed', 'TBchg75', 'TBchgmax']]
        return base_names