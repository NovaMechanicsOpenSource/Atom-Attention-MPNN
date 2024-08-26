from typing import List
import torch
from tqdm import tqdm

from .data import MoleculeDataLoader, MoleculeDataset
from .model import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False) -> List[List[float]]:
    model.eval()
    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()

        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
    return preds