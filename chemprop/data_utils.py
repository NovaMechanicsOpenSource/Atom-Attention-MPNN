from collections import OrderedDict
import csv
from logging import Logger
import pickle
import random
from random import Random
from typing import List, Optional, Set, Tuple, Union
import os
from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split
from .args import PredictArgs, TrainArgs

def preprocess_smiles_columns(path: str,
                              smiles_columns: Optional[Union[str, List[Optional[str]]]],
                              number_of_molecules: int = 1) -> List[Optional[str]]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES.
    """
    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if not isinstance(smiles_columns, list):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError(
                    'Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError(
                    'Provided smiles_columns do not match the header of data file.')
    return smiles_columns


def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None,
                   target_columns: List[str] = None) -> List[str]:
    if target_columns is not None:
        return target_columns
    columns = get_header(path)
    target_names = [column for column in columns if column not in smiles_columns]
    return target_names


def get_header(path: str) -> List[str]:
    with open(path) as f:
        header = next(csv.reader(f))
    return header


def get_smiles(path: str,
               smiles_columns: Union[str, List[str]] = None,
               header: bool = True,
               flatten: bool = False
               ) -> Union[List[str], List[List[str]]]:
    """Returns the SMILES from a data CSV file."""
    if smiles_columns is not None and not header:
        raise ValueError('If smiles_column is provided, the CSV file must have a header.')

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    with open(path) as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            smiles_columns = 0

        smiles = [[row[c] for c in smiles_columns] for row in reader]

    if flatten:
        smiles = [smile for smiles_list in smiles for smile in smiles_list]
    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """Filters out invalid SMILES."""
    return MoleculeDataset([datapoint for datapoint in tqdm(data)
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol)])

def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             args: Union[TrainArgs, PredictArgs] = None,
             features_generator: List[str] = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             skip_none_targets: bool = False) -> MoleculeDataset:
    debug = logger.debug if logger is not None else print

    if args is not None:
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        features_generator = features_generator if features_generator is not None else args.features_generator
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(
            path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')
    features_data = None

    with open(path) as f:
        reader = csv.DictReader(f)

        if target_columns is None:
            target_columns = get_task_names(path=path, smiles_columns=smiles_columns, target_columns=target_columns)

        all_smiles, all_targets, all_rows, all_features = [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]
            targets = [float(row[column]) if row[column] !=
                       '' else None for column in target_columns]
            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)
            if features_data is not None:
                all_features.append(features_data[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                row = all_rows[i] if store_row else None,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                               total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(
                f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_MPN(path: str,
             smiles_columns: Union[str, List[str]] = None,
             skip_invalid_smiles: bool = True,
             args: Union[TrainArgs, PredictArgs] = None,
             features_generator: List[str] = None,
             max_data_size: int = None,
             logger: Logger = None) -> MoleculeDataset:
    debug = logger.debug if logger is not None else print

    if args is not None:
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        features_generator = features_generator if features_generator is not None else args.features_generator
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(
            path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')
    features_data = None

    with open(path) as f:
        reader = csv.DictReader(f)

        all_smiles, all_features = [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            all_smiles.append(smiles)
            if features_data is not None:
                all_features.append(features_data[i])

            if len(all_smiles) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                features_generator=features_generator,
                features=all_features[i] if features_data is not None else None
            ) for i, (smiles) in tqdm(enumerate(all_smiles),
                                               total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(
                f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """Splits data into training, validation, and test splits."""
    if not (len(sizes) == 3 and sum(sizes) == 1):
        raise ValueError(
            'Valid split sizes must sum to 1 and must have three sizes: train, validation, and test.')

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'cv':
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(
                'Number of folds for cross-validation must be between 2 and len(data), inclusive.')

        random = Random(0)

        indices = np.repeat(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = (seed) % num_folds
        #val_index = (seed + 1) % num_folds

        train, val, test, val_test = [], [], [], []
        for d, index in zip(data, indices):
            if index == test_index:
                val_test.append(d)
                val_ts_indices = list(range(len(val_test)))
                random = Random(seed)
                random.shuffle(val_ts_indices)
                val = [val_test[i] for i in val_ts_indices[:int(0.5*len(val_test))]]
                test = [val_test[i] for i in val_ts_indices[int(0.5*len(val_test)):]]
            else:
                train.append(d)

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)



    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError(
                'Split indices must have three splits: train, validation, and test')

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index and sizes[2] != 0:
            raise ValueError('Test size must be zero since test set is created separately '
                             'and we want to put all other data in train and validation')

        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                # in case we're loading indices from python2
                all_fold_indices = pickle.load(f, encoding='latin1')

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices]
                 for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    targets = data.targets()
    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError(
                'Classification dataset must only contains 0s and 1s.')
        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes