from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR


from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .args import TrainArgs
from .constants import MODEL_FILE_NAME
from .data_utils import get_class_sizes, get_data, split_data
from .data import MoleculeDataLoader, MoleculeDataset
from .model import MoleculeModel
from .nn_utils import param_count
from .utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, load_checkpoint_MPN, makedirs, save_checkpoint, save_smiles_splits
from .plot_auc import plot_roc_curve


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    torch.manual_seed(args.pytorch_seed)

    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             smiles_columns=args.smiles_columns, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            smiles_columns=args.smiles_columns, logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=(0.8, 0.0, 0.2),
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    class_sizes = get_class_sizes(data)
    debug('Class sizes')
    for i, task_class_sizes in enumerate(class_sizes):
        debug(f'{args.task_names[i]} '
                f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    scaler = None

    loss_func = get_loss_func(args)

    train_smiles, train_targets = train_data.smiles(), train_data.targets()
    val_smiles, val_targets = val_data.smiles(), val_data.targets()
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_train_preds = np.zeros((len(train_smiles), args.num_tasks))
    sum_val_preds = np.zeros((len(val_smiles), args.num_tasks))
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed)

    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        elif args.pretrained_checkpoint is not None:
            model = MoleculeModel(args)
            debug('here import model')
            state_dict = torch.load(args.pretrained_checkpoint, map_location=args.device)
            #model.load_my_state_dict(state_dict)
            model.encoder = load_checkpoint_MPN(args.pretrained_checkpoint, logger=logger)
            print("Loaded pre-trained model with success.")
            #for param in model.parameters():
            #    param.requires_grad = False
            # Then, enable requires_grad for FFN parameters
            #for param in model.ffn.parameters():
            #    param.requires_grad = True

            # Optionally, you may print to verify which parameters have requires_grad=True
            for name, param in model.named_parameters():
                print("HEEEEERE", name, param.requires_grad)

        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)
            debug('here import model, intitialize all weights????')

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler, args)

        optimizer = build_optimizer(model, args)
        scheduler = build_lr_scheduler(optimizer, args)

        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer)
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores, _, _ = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                scaler=scaler,
                logger=logger)

            for metric, scores in val_scores.items(): 
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)


            avg_val_score = np.nanmean(val_scores[args.metric])


            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME),
                                model, scaler, features_scaler, args)

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        args.features_size = data.features_size()

        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        train_preds = predict(
            model=model,
            data_loader=train_data_loader,
            scaler=scaler)

        if len(train_preds) != 0:
            sum_train_preds += np.array(train_preds)

        val_preds = predict(
            model=model,
            data_loader=val_data_loader,
            scaler=scaler)

        if len(val_preds) != 0:
            sum_val_preds += np.array(val_preds)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler)

        test_scores, _, _ = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            logger=logger)

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)
        writer.close()

    avg_val_preds = (sum_val_preds / args.ensemble_size).tolist()
    ensemble_scores_val, df_val_preds, df_val_tar = evaluate_predictions(
        preds=avg_val_preds,
        targets=val_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        logger=logger)

    if 'auc' in args.metrics:
        plot_roc_curve(val_targets, avg_val_preds, os.path.join(args.save_dir, 'val_roc_curve.png'))

    for metric, scores in ensemble_scores_val.items():
        avg_ensemble_val_score = np.nanmean(scores)
        info(f'Ensemble val {metric} = {avg_ensemble_val_score:.6f}')

    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    ensemble_scores_ts, df_ts_preds, df_ts_tar = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        logger=logger)


    if 'auc' in args.metrics:
        plot_roc_curve(test_targets, avg_test_preds, os.path.join(args.save_dir, 'test_roc_curve.png'))

    for metric, scores in ensemble_scores_ts.items():
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores_val, ensemble_scores_ts


