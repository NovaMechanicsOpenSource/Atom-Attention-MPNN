from typing import List, Union
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .attention import MultiAtomAttention, SublayerConnection
from .args import TrainArgs
from .featurization import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from .nn_utils import index_select_ND, get_activation_function, initialize_weights
from .attention_visualization import visualize_atom_attention


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout

        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        self.atom_attention = args.atom_attention

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.act_func = get_activation_function(args.activation)

        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.atom_attention:
            self.atom_attention_block = MultiAtomAttention(args)
            self.atom_residual = SublayerConnection(dropout=self.dropout)

    def forward(self,
                mol_graph: BatchMolGraph,
                viz_dir: str = None) -> torch.FloatTensor:
        """Encodes a batch of molecular graphs."""

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(
            self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)

        message = self.act_func(input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)

            else:
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b

        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)

        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)


                if self.atom_attention:
                    mol_vec, att_a_w = self.atom_attention_block(cur_hiddens)
                    mol_vec = self.atom_residual(cur_hiddens, mol_vec)
                    if viz_dir:
                        visualize_atom_attention(viz_dir, mol_graph.smiles_batch[i], a_size, att_a_w)
                else:
                    mol_vec = cur_hiddens

                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs, atom_fdim: int = None, bond_fdim: int = None):
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device

        #initialize_weights(self)

        if self.features_only:
            return

        self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim) for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[BatchMolGraph]]) -> torch.FloatTensor:
        """Encodes a batch of molecules."""

        if type(batch[0]) != BatchMolGraph:
            print('type(batch[0]) != BatchMolGraph')
            batch = [mol2graph(b) for b in batch]

        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)
        return output

    def viz_attention(self,
                      batch: Union[List[str], BatchMolGraph],
                      viz_dir: str = None):
        encodings = [enc(ba, viz_dir=viz_dir) for enc, ba in zip(self.encoder, batch)]