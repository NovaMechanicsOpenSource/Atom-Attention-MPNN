import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .args import TrainArgs
from .nn_utils import get_activation_function
from .attention_visualization import visualize_atom_attention


class MultiAtomAttention(nn.Module):
    def __init__(self, args: TrainArgs):
        super(MultiAtomAttention, self).__init__()
        self.atom_attention = args.atom_attention
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.dropout = args.dropout
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.normalize_matrices = args.normalize_matrices
        self.device = args.device
        self.num_heads = args.num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5

        self.W_a_q = nn.Linear(
            self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_k = nn.Linear(
            self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_v = nn.Linear(
            self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_o = nn.Linear(
            self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

    def forward(self, cur_hiddens):
        """Calculate the atom-level attention of a molecule with Transformer in the readout phase."""
        # cur_hidden (1 mol): num_atoms x hidden_size
        cur_hiddens_size = cur_hiddens.size()

        # (num_atoms, num_head, att_size)
        a_q = self.W_a_q(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        # (num_atoms, num_head, att_size)
        a_k = self.W_a_k(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        # (num_atoms, num_head, att_size)
        a_v = self.W_a_v(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_q = a_q.transpose(0, 1)  # (num_head, num_atoms, att_size)
        # (num_head, att_size, num_atoms)
        a_k = a_k.transpose(0, 1).transpose(1, 2)
        a_v = a_v.transpose(0, 1)  # (num_head, num_atoms, att_size)

        att_a_w = torch.matmul(a_q, a_k)  # (num_head, num_atoms, num_atoms)

        # (num_head, num_atoms, num_atoms)
        att_a_w = F.softmax(att_a_w * self.scale_factor, dim=2)
        att_a_h = torch.matmul(att_a_w, a_v)  # (num_head, num_atoms, att_size)
        att_a_h = self.act_func(att_a_h)
        att_a_h = self.dropout_layer(att_a_h)

        # (num_atom, num_head, att_size)
        att_a_h = att_a_h.transpose(0, 1).contiguous()
        # (num_atom, hidden_size)
        att_a_h = att_a_h.view(cur_hiddens_size[0], self.num_heads * self.att_size)
        att_a_h = self.W_a_o(att_a_h)  # (num_atoms, hidden_size)
        assert att_a_h.size() == cur_hiddens_size

        att_a_h = att_a_h.unsqueeze(dim=0)
        att_a_h = self.norm(att_a_h)

        mol_vec = (att_a_h).squeeze(dim=0)  # (num_atoms, hidden_size)
        return mol_vec, torch.mean(att_a_w, axis=0)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, original, attention):
        """Apply residual connection to any sublayer with the same size."""
        return original + self.dropout(attention)