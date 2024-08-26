from typing import List, Union
import logging
import os
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from .constants import PRETRAINED_MODEL_FILE_NAME
from .mpn import MPN
from .args import TrainArgs
from .featurization import BatchMolGraph
from .nn_utils import get_activation_function, initialize_weights, initialize_weights_tf


class MoleculeModel(nn.Module):
    def __init__(self, args: TrainArgs, featurizer: bool = False):
        super(MoleculeModel, self).__init__()
        self.featurizer = featurizer
        self.pretrained = False if args.pretrained_checkpoint is None else True
        self.output_size = args.num_tasks
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)
        self.create_ffn(args)
        
        if self.pretrained:
            print('initialize only ffn')
            initialize_weights_tf(self)
        else:
            print('initialize all model weights')
            initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """Creates the message passing encoder for the model."""
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs) -> None:
        """Creates the feed-forward layers for the model."""
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.ffn_hidden_size)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(args.ffn_hidden_size, self.output_size)])

        self.ffn = nn.Sequential(*ffn)

    def featurize(self,
                  batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                  features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """Computes feature vectors of the input by running the model except for the last layer. (MPNEncoder + FFN[:-1])"""
        return self.ffn[:-1](self.encoder(batch, features_batch))

    def fingerprint(self,
                    batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                    features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes the fingerprint vectors of the input molecules by passing the inputs through the MPNN and returning
        the latent representation before the FFNN. (MPNEncoder)
        return: The fingerprint vectors calculated through the MPNN.
        """
        return self.encoder(batch, features_batch)

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """Runs the :class:`MoleculeModel` on input."""
        if self.featurizer:
            return self.featurize(batch, features_batch)
        
        if self.pretrained:
            self.encoder.eval()
            with torch.no_grad():
                output = self.encoder(batch)
        else:
            output = self.encoder(batch)
            self.encoder.train()

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)
    

        output = self.ffn(output)
        if not self.training:
            output = self.sigmoid(output)

        return output

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)
