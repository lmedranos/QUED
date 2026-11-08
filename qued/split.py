import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Sequence
 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

from qued.data import QUEDDataset

class QUEDSplitter(ABC):
    """ Defines shared 'group molecules into buckets, then
        pack buckets into train/val/test' logic. """
    def __init__(self, dataset: QUEDDataset, seed: int = 0):
        self.dataset = dataset
        self._n = len(dataset) 
        self.seed = seed
        self._groups: Optional[List[List[int]]] = None
        self._group_id: Optional[np.ndarray] = None

    @abstractmethod # 
    def get_groups(self) -> List[List[int]]:
        """ 
        Return a list of groups; each inner list is a set of dataset
        indices that must end up in the same split. 
        """
        # Each subclass implements its own
        raise NotImplementedError

    def _ensure_groups(self):
        if self._groups is None:
            self._groups = self.get_groups()
            # initialize as empty
            group_id = np.empty(self._n, dtype=int)
            # assign a group ID to every index in self._groups
            for gid, members in enumerate(self._groups):
                for i in members:
                    group_id[i] = gid
            self._group_id = group_id

    @property
    def group_id(self) -> np.ndarray:
        """ Per-molecule group label, aligned with dataset indices. 
        Pass this as `groups=` to sklearn's GroupKFold for HPO CV 
        consistent with this split."""
        self._ensure_groups()
        return self._group_id
    
    def split(self, n_train: Optional[int] = None, 
              frac_train: float = 0.7, 
              frac_val: float = 0.0,
              frac_test: float = 0.3, 
              sort_type='by_size'):
        """ 
        Bin-packs whole groups into train/val/test so groups are never    
        split across partitions. Returns (train_idx, val_idx, test_idx). """

        assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
        self._ensure_groups()

        groups = list(self._groups)
        rng = random.Random(self.seed)

        if sort_type=='by_size':
            threshold = max(1, int(self._n * 0.01))
            big, small = [], []
            for g in groups:
                (big if len(g) > threshold else small).append(g)
            rng.shuffle(big)
            rng.shuffle(small)
            groups = big + small
        elif sort_type=='random':
            rng.shuffle(groups)
        else:
            groups.sort(key=len, reverse=True)

        if n_train is not None:
            n_val = 0
        else:
            n_train, n_val = int(frac_train * self._n), int(frac_val * self._n)
        train_idx, val_idx, test_idx = [], [], []
        for g in groups:
            if len(train_idx) + len(g) <= n_train:
                train_idx.extend(g)
            elif len(val_idx) + len(g) <= n_val:
                val_idx.extend(g)
            else:
                test_idx.extend(g)
        return train_idx, val_idx, test_idx

### Split train/validation
def split_subset(groups, indices, frac_train=0.9, seed=0):
    """Split a subset into train/validation while keeping groups intact."""

    rng = random.Random(seed)
    indices = set(indices)

    # Keep only members belonging to the development set
    groups = [
        [i for i in group if i in indices]
        for group in groups
    ]
    groups = [g for g in groups if g]

    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_train = int(frac_train * len(indices))
    train_idx, val_idx = [], []

    for group in groups:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        else:
            val_idx.extend(group)

    return train_idx, val_idx

### SCAFFOLD METHOD ###
class ScaffoldSplitter(QUEDSplitter):
    """Bemis-Murcko scaffold split, at the molecule level."""

    def __init__(self, dataset, generic: bool = False, subdivide_oversized=False, seed: int = 0):
        super().__init__(dataset, seed=seed)
        # generic strips away atom identity and bond order 
        # from the ring/linker skeleton
        self.generic = generic
        # Some scaffolds are incredibly large
        # split them by similarity of fingerprints
        self.subdivide_oversized = subdivide_oversized
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        self.dataset._scaffold_cache = {}

    # ---------- scaffold utilities ----------
    def _get_scaffold(self, idx: int) -> str:
        if idx in self.dataset._scaffold_cache:
            return self.dataset._scaffold_cache[idx]
        mol = self.dataset.get_mol(idx)
        if mol is None:
            scaffold = ""
        else:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if self.generic:
                scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            scaffold = Chem.MolToSmiles(scaffold_mol)
        self.dataset._scaffold_cache[idx] = scaffold
        return scaffold

    def _morgan_fingerprint(self, idx: int) -> np.ndarray:
        mol = self.dataset.get_mol(idx)
        fp = self.mfpgen.GetFingerprint(mol)
        return np.array(fp, dtype=float)

    def _subdivide_oversized_groups(self, groups: List[List[int]]) -> List[List[int]]:
        total_n = len(self.dataset)
        max_size = max(1, int(0.05 * total_n))
        rng = np.random.default_rng(self.seed)
 
        result = []
        queue = list(groups)
        while queue:
            g = queue.pop()
            if len(g) <= max_size:
                result.append(g)
                continue

            X = np.stack([self._morgan_fingerprint(i) for i in g])
            # Bisect into 2 groups via FPS
            start = int(rng.integers(len(g)))
            min_dist = np.linalg.norm(X - X[start], axis=1)
            other = int(np.argmax(min_dist))
            landmarks = [start, other]
            cluster_id, _ = pairwise_distances_argmin_min(X, X[landmarks])

            sub: Dict[int, List[int]] = defaultdict(list)
            for local_idx, cid in enumerate(cluster_id):
                sub[int(cid)].append(g[local_idx])  # map back to real dataset indices
            subgroups = list(sub.values())

            # Safety guard: if the "split" didn't actually separate
            # anything (e.g. near-identical fingerprints put every point
            # in one bucket), stop recursing on this group instead of
            # looping forever on an unsplittable group.
            if len(subgroups) < 2 or max(len(s) for s in subgroups) == len(g):
                print(f"[warning] could not further subdivide a group of "
                      f"{len(g)} molecules with distinguishable fingerprints - "
                      f"keeping it as one oversized group.")
                result.append(g)
                continue
 
            queue.extend(subgroups)
 
        return result
 
    def get_groups(self) -> List[List[int]]:
        scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(self.dataset)):
            scaffold = self._get_scaffold(i)
            scaffold_to_indices[scaffold].append(i)
        groups = list(scaffold_to_indices.values())

        if self.subdivide_oversized:
            groups = self._subdivide_oversized_groups(groups)
        return groups

### PUBLISHED QUED SPLITTING METHOD ###
class LegacyFPSSplitter(QUEDSplitter):
    """ Test """
    def __init__(self, dataset, X_features: np.ndarray, seed: int = 0):
        super().__init__(dataset, seed=seed)
        self._order: Optional[np.ndarray] = None
        self._X = X_features

    @staticmethod
    def _fps_order(X: np.ndarray) -> np.ndarray:
        # Full farthest-point ranking of every row in X
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        n = X.shape[0]
        center = X.mean(axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        order = np.zeros(n, dtype=int)
        order[0] = int(np.argmax(distances))
 
        for i in range(1, n):
            current_point = X[order[i - 1]]
            new_distances = np.linalg.norm(X - current_point, axis=1)
            distances = np.minimum(distances, new_distances)
            order[i] = int(np.argmax(distances))
        return order # np.flip(order)

    def get_groups(self, indices: Optional[Sequence[int]] = None,) -> List[List[int]]:
        if indices is None:
            indices = np.arange(len(self._X))
        else:
            indices = np.asarray(indices, dtype=int)

        # Compute FPS using only the selected molecules
        order = self._fps_order(self._X[indices])
        # Map local FPS indices back to original dataset indices
        return [[int(indices[i])] for i in order]

    def split(
        self,
        n_train: Optional[int] = None,
        frac_train: float = 0.7,
        frac_val: float = 0.0,
        frac_test: float = 0.3,
        sort_type='none'
    ) -> Tuple[List[int], List[int], List[int]]:
        # preserves FPS order instead of shuffling groups (shuffling would
        # defeat the entire purpose of an exact legacy reproduction).
        return super().split(n_train, frac_train, frac_val, frac_test, sort_type='none')

### FPS+CLUSTER METHOD ###
class FPSClusterSplitter(QUEDSplitter):
    """ 
    FPS-derived cluster split. 
    Farthest-point sampling picks n_clusters landmark points; 
    every molecule is then assigned to its nearest landmark. 
    Groups = molecules sharing a landmark, so entire regions of
    feature space are held out together.
    Args:
    - 
    """

    def __init__(self, dataset, X_features: np.ndarray, 
                 n_clusters: int, seed: int=0):
        super().__init__(dataset, seed=seed)
        self.n_clusters = n_clusters
        self._features = X_features

    def _farthest_point_sampling(self, X: np.ndarray, n_points: int) -> np.ndarray:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        selected = [int(rng.integers(n))]
        min_dist = np.linalg.norm(X - X[selected[0]], axis=1)
        for _ in range(n_points - 1):
            next_idx = int(np.argmax(min_dist))
            selected.append(next_idx)
            new_dist = np.linalg.norm(X - X[next_idx], axis=1)
            min_dist = np.minimum(min_dist, new_dist)
        return np.array(selected)

    def get_groups(self) -> List[List[int]]:
        X = self._features
        landmarks = self._farthest_point_sampling(X, self.n_clusters)
        cluster_id, _ = pairwise_distances_argmin_min(X, X[landmarks])
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, c in enumerate(cluster_id):
            clusters[int(c)].append(i)
        return list(clusters.values())

