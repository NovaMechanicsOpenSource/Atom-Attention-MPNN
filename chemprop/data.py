import threading
from random import Random
from typing import Dict, Iterator, List, Optional, Union
from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from .features_generators import get_features_generator
from .featurization import BatchMolGraph, MolGraph


CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}

def cache_mol() -> bool:
    return CACHE_MOL

def set_cache_mol(cache_mol: bool) -> None:
    global CACHE_MOL
    CACHE_MOL = cache_mol

class MoleculeDatapoint:
    def __init__(self, smiles: List[str], targets: List[Optional[float]] = None, row:OrderedDict = None, features: np.ndarray = None, features_generator: List[str] = None):

        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.smiles = smiles
        self.targets = targets
        self.row = row
        self.features = features
        self.features_generator = features_generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                for m in self.mol:
                    if m is not None and m.GetNumHeavyAtoms() > 0:
                        self.features.extend(features_generator(m))
                    elif m is not None and m.GetNumHeavyAtoms() == 0:
                        self.features.extend(np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))

            self.features = np.array(self.features)

        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        self.raw_features, self.raw_targets = self.features, self.targets

    @property
    def mol(self) -> List[Chem.Mol]:
        mol = [SMILES_TO_MOL.get(s, Chem.MolFromSmiles(s)) for s in self.smiles]
        if cache_mol():
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m
        return mol

    @property
    def number_of_molecules(self) -> int:
        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

    def extend_features(self, features: np.ndarray) -> None:
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        self.features, self.targets = self.raw_features, self.raw_targets


class MoleculeDataset(Dataset):
    def __init__(self, data: List[MoleculeDatapoint]):
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]]]:
        if flatten:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self, randomized: bool = False) -> List[BatchMolGraph]:

        if randomized:
            self._batch_graph = None

        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if randomized:
                        #print('Randomizeddddd')
                        mol_graph = MolGraph(m, mask_atoms=True)
                    else:
                        #print('Not randomized')
                        mol_graph = MolGraph(m)
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
        return self._batch_graph


    def features(self) -> List[np.ndarray]:
        """Returns the features associated with each molecule (if they exist)."""
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """ Returns the targets associated with each molecule."""
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        """Returns the number of prediction tasks."""
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """Returns the size of the additional features vector associated with the molecules."""
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_features: bool = False) -> StandardScaler:

        if len(self._data) == 0 or (self._data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        for d in self._data:
            d.set_features(self._scaler.transform(d.raw_features.reshape(1, -1))[0])

        return self._scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self._data[item]


class MoleculeSampler(Sampler):
    def __init__(self, dataset: MoleculeDataset, class_balance: bool = False, shuffle: bool = False, seed: int = 0):
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle
        self._random = Random(seed)
        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices),len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    data = MoleculeDataset(data)
    data.batch_graph()
    return data


class MoleculeDataLoader(DataLoader):
    def __init__(self, dataset: MoleculeDataset, batch_size: int = 50, num_workers: int = 8, 
                class_balance: bool = False, shuffle: bool = False, seed: int = 0):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'
            self._timeout = 3600

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed)
        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout)

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """Returns the targets associated with each molecule."""
        if self._class_balance or self._shuffle:
            raise ValueError(
                'Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()