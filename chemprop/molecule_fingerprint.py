import csv
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from .args import PredictArgs, TrainArgs
from .data import MoleculeDataLoader, MoleculeDataset
from .data_utils import get_data
from .utils import load_args, load_checkpoint, makedirs, timeit, load_scalers, update_prediction_args
from .model import MoleculeModel

@timeit()
def molecule_fingerprint(args: PredictArgs) -> List[List[Optional[float]]]:
    """Loads data and a trained model and uses the model to encode fingerprint vectors for the data."""

    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])

    update_prediction_args(predict_args=args, train_args=train_args, validate_feature_sources=False)
    args: Union[PredictArgs, TrainArgs]

    print('Loading data')
    full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], skip_invalid_smiles=False,
                            args=args, store_row=True)

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

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    print(f'Encoding smiles into a fingerprint vector from a single model')
    if len(args.checkpoint_paths) != 1:
        raise ValueError("Fingerprint generation only supports one model, cannot use an ensemble")

    model = load_checkpoint(args.checkpoint_paths[0], device=args.device)
    scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(args.checkpoint_paths[0])

    if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
        test_data.reset_features_and_targets()
        if args.features_scaling:
            test_data.normalize_features(features_scaler)
        if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        if train_args.bond_feature_scaling and args.bond_features_size > 0:
            test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

    model_preds = model_fingerprint(
        model=model,
        data_loader=test_data_loader)

    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(model_preds)
    makedirs(args.preds_path, isfile=True)

    total_hidden_size = args.hidden_size * args.number_of_molecules
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = model_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * total_hidden_size

        fingerprint_columns=[f'fp_{i}' for i in range(total_hidden_size)]
        for i in range(len(fingerprint_columns)):
            datapoint.row[fingerprint_columns[i]] = preds[i]

    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=args.smiles_columns+fingerprint_columns,extrasaction='ignore')
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)

    return model_preds

def model_fingerprint(model: MoleculeModel,
            data_loader: MoleculeDataLoader) -> List[List[float]]:
    """Encodes the provided molecules into the latent fingerprint vectors, according to the provided model."""
    
    model.eval()
    fingerprints = []

    for batch in tqdm(data_loader, disable=False, leave=False):
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()
        with torch.no_grad():
            batch_fp = model.fingerprint(mol_batch, features_batch, atom_descriptors_batch)

        batch_fp = batch_fp.data.cpu().tolist()
        fingerprints.extend(batch_fp)
    return fingerprints

def chemprop_fingerprint() -> None:
    molecule_fingerprint(args=PredictArgs().parse_args())