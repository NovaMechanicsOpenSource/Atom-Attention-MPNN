import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import Logger
from argparse import Namespace
import os
from typing import Dict, List
import numpy as np
from tensorboardX import SummaryWriter
import tqdm
from typing import Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .train import train
from .args import TrainArgs
from .constants import PRETRAINED_MODEL_FILE_NAME
from .data_utils import get_data_MPN, split_data
from .data import MoleculeDataLoader, MoleculeDataset
from .mpn import MPN
from .nt_xent import NTXentLoss
from .utils import build_optimizer, build_lr_scheduler, makedirs, save_checkpoint

def step_(model, inp, tar, criterion):
    out1 = model(inp)
    out2 = model(tar)
    out1 = F.normalize(out1, dim=1)
    out2 = F.normalize(out2, dim=1)
    loss = criterion(out1, out2)
    return loss

def pre_train(args: TrainArgs, logger: Logger = None) -> Dict[str, List[float]]:
    data = get_data_MPN(path=args.data_path, args=args, smiles_columns=args.smiles_columns, logger=logger)
    args.train_data_size = len(data)
    train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.9, 0.1, 0.0), seed=args.seed, num_folds=args.num_folds, args=args, logger=logger)
    train_dataloader = MoleculeDataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, seed=args.seed)
    valid_dataloader = MoleculeDataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, seed=args.seed)
    criterion = NTXentLoss(device=args.device, batch_size=args.batch_size, temperature=0.1, use_cosine_similarity=True)
    for model_idx in range(args.ensemble_size):
        model = MPN(args)
        model = model.to(args.device)
        optimizer = build_optimizer(model, args)
        scheduler = build_lr_scheduler(optimizer, args)
        makedirs(args.save_dir)
        best_valid_loss = np.inf
        patience=5
        for epoch in range(args.epochs):
            print('Epoch', epoch)
            loss = _train(model=model, data_loader=train_dataloader, args=args, optimizer=optimizer, scheduler=scheduler, criterion=criterion)
            if (epoch+1)%2==0:
                valid_loss = _validate(model=model, data_loader=valid_dataloader, args=args, criterion=criterion)
                print(epoch, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_checkpoint_here(os.path.join(args.save_dir, PRETRAINED_MODEL_FILE_NAME), model, args)
                    patience = 5
                else:
                    patience-=1
                    if patience == 0 :
                        break

def _train(model: MPN, data_loader: MoleculeDataLoader, args: TrainArgs,optimizer: Optimizer,scheduler: _LRScheduler, criterion: Callable) -> List[List[float]]:
    counter = 0
    train_loss = 0.0
    for batch in tqdm.tqdm(data_loader, total=len(data_loader), leave=False):
        batch: MoleculeDataset
        optimizer.zero_grad()
        mol_batch = batch.batch_graph()
        rand_mol_batch = batch.batch_graph(randomized=True)
        loss = step_(model, rand_mol_batch, mol_batch, criterion)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        counter += 1
    train_loss /= counter
    return train_loss
        
def _validate(model: MPN, data_loader: MoleculeDataLoader, args: TrainArgs, criterion: Callable) -> List[List[float]]:
    # validation steps
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        counter = 0
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), leave=False):
            batch: MoleculeDataset
            mol_batch = batch.batch_graph()
            rand_mol_batch = batch.batch_graph(randomized=True)    
            loss = step_(model, rand_mol_batch, mol_batch, criterion)
            valid_loss += loss.item()
            counter += 1
        valid_loss /= counter
    model.train()
    return valid_loss


def save_checkpoint_here(path: str, model: MPN, args: TrainArgs = None) -> None:
    if args is not None:
        args = Namespace(**args.as_dict())
    state = {
        'args': args,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)
