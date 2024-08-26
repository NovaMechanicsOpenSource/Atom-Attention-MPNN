from typing import List

import torch
from tqdm import tqdm

from .data import MoleculeDataLoader, MoleculeDataset
from .scaler import StandardScaler
from .model import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """Makes predictions on a dataset using an ensemble of models."""
    model.eval()
    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
