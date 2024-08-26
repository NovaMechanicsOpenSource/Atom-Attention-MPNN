from collections import OrderedDict
import csv
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
from .args import PredictArgs, TrainArgs

from .predict import predict
from .data import MoleculeDataLoader, MoleculeDataset
from .data_utils import get_data
from .utils import load_args, load_checkpoint, makedirs, timeit, update_prediction_args,load_scalers


@timeit()
def make_predictions_and_viz(args: PredictArgs) -> List[List[Optional[float]]]:
    # Load checkpoint_paths (here we load the folder where the saved model is)
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]
    # test_path is the csv file with one column with smiles
    print('Loading data')
    full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[],skip_invalid_smiles=False, store_row=True, args=args)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    sum_preds = np.zeros((len(test_data), num_tasks))

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        model = load_checkpoint(checkpoint_path, device=args.device)
        features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(checkpoint_path)

        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        model_preds = predict(model=model, data_loader=test_data_loader)

        sum_preds += np.array(model_preds)

    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    makedirs(args.preds_path, isfile=True)

    for full_index, datapoint in enumerate(full_data):
        print('datapoint', datapoint)
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
        print('preds', preds)

        for pred_name, pred in zip(task_names, preds):
            datapoint.row[pred_name] = pred

    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)

    mpn = model.encoder
    for it, batch in enumerate(tqdm(test_data_loader, leave=False)):
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        mpn.viz_attention(mol_batch, args.viz_dir)
    print('All vizualisations done!')    
    return avg_preds


def chemprop_predict() -> None:
    make_predictions_and_viz(args=PredictArgs().parse_args())