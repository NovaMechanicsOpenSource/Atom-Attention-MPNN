from collections import defaultdict
import logging
from typing import Dict, List
import pandas as pd

from .predict import predict
from .data import MoleculeDataLoader
from .scaler import StandardScaler
from .model import MoleculeModel
from .utils import get_metric_func, confusion_matrix_


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """Evaluates predictions using a metric function after filtering out invalid targets."""
    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    val_pred = []
    val_tar = []
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    results = defaultdict(list)
    for i in range(num_tasks):
        nan = False
        if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
            nan = True
            info('Warning: Found a task with targets all 0s or all 1s')
        if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
            nan = True
            info('Warning: Found a task with predictions all 0s or all 1s')

        if nan:
            for metric in metrics:
                results[metric].append(float('nan'))
            continue

        cf = confusion_matrix_(valid_targets[i], valid_preds[i])
        print('confusion_matrix_ ', cf)
            

        if len(valid_targets[i]) == 0:
            continue

        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

        val_pred.append(valid_preds[i])
        val_tar.append(valid_targets[i])
        df1 = pd.DataFrame(val_pred)
        df2 = pd.DataFrame(val_tar)

    results = dict(results)

    return results, df1, df2


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions."""
    preds = predict(model=model, data_loader=data_loader, scaler=scaler)
    results, df1, df2 = evaluate_predictions(preds=preds, targets=data_loader.targets, num_tasks=num_tasks, metrics=metrics, logger=logger)
    return results, df1, df2
