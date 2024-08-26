from collections import defaultdict
import csv
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .run_training import run_training
from .args import TrainArgs
from .constants import VAL_SCORES_FILE_NAME, TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from .data_utils import get_data, get_task_names
from .data import MoleculeDataset
from .utils import create_logger, makedirs, timeit
from .featurization import set_extra_atom_fdim, set_extra_bond_fdim

@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs,
                   train_func: Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.
    """
    # logger: quite and verbose file
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')
    debug('Args')
    debug(args)
    makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, 'args.json'))

    debug('Loading data')
    data = get_data(
        path=args.data_path, args=args,
        smiles_columns=args.smiles_columns, logger=logger, skip_none_targets=True)
    args.features_size = data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
        args.ffn_hidden_size += args.atom_descriptors_size
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    all_scores_val = defaultdict(list)
    all_scores_ts = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()
        model_scores_val, model_scores_ts = train_func(args, data, logger)
        for metric, scores in model_scores_val.items():
            all_scores_val[metric].append(scores)
        for metric, scores in model_scores_ts.items():
            all_scores_ts[metric].append(scores)

    all_scores_val = dict(all_scores_val)
    all_scores_ts = dict(all_scores_ts)
    for metric, scores in all_scores_val.items():
        all_scores_val[metric] = np.array(scores)
    for metric, scores in all_scores_ts.items():
        all_scores_ts[metric] = np.array(scores)

    info(f'{args.num_folds}-fold cross validation')

    for fold_num in range(args.num_folds):
        for metric, scores in all_scores_val.items():
            info(f'\tSeed {init_seed + fold_num} ==> valid {metric} = {np.nanmean(scores[fold_num]):.6f}')

    for metric, scores in all_scores_val.items():
        avg_scores = np.nanmean(scores, axis=1)
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'Overall validation {metric} = {mean_score:.6f} +/- {std_score:.6f}')

    for fold_num in range(args.num_folds):
        for metric, scores in all_scores_ts.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

    for metric, scores in all_scores_ts.items():
        avg_scores = np.nanmean(scores, axis=1)
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

    with open(os.path.join(save_dir, VAL_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)
        for task_num, task_name in enumerate(args.task_names):
            row = [task_name]
            for metric, scores in all_scores_val.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
    avg_scores_val = np.nanmean(all_scores_val[args.metric], axis=1)
    mean_score_val, std_score_val = np.nanmean(avg_scores_val), np.nanstd(avg_scores_val)

    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)
        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)
        for task_num, task_name in enumerate(args.task_names):
            row = [task_name]
            for metric, scores in all_scores_ts.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)
    avg_scores_ts = np.nanmean(all_scores_ts[args.metric], axis=1)
    mean_score_ts, std_score_ts = np.nanmean(avg_scores_ts), np.nanstd(avg_scores_ts)

    # Optionally merge and save test preds
    if args.save_preds:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                               for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score_val, std_score_val, mean_score_ts, std_score_ts


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.
    This is the entry point for the command line command :code:`chemprop_train`.
    """
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)