import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt

from .nn_utils import index_select_ND


def visualize_atom_attention(viz_dir: str,
                             smiles: str,
                             num_atoms: int,
                             attention_weights: torch.FloatTensor):
    """
    Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch
    :param viz_dir: Directory in which to save attention map figures.
    :param smiles: Smiles string for molecule.
    :param num_atoms: The number of atoms in this molecule.
    :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
    """
    if type(smiles) == str:
        mol_name = smiles
        print('Saving {0} ({1} atoms)'.format(smiles, num_atoms))
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
        mol_name = Chem.MolToSmiles(mol)
        print('Saving Similarity map of molecule: {0} ({1} atoms)'.format(mol_name, num_atoms))

    smiles_viz_dir = viz_dir
    os.makedirs(smiles_viz_dir, exist_ok=True)
    atomSum_weights = np.zeros(num_atoms)

    for a in range(num_atoms):
        a_weights = attention_weights[a].cpu().data.numpy()
        atomSum_weights += a_weights

    Amean_weight = atomSum_weights / num_atoms

    nanMean = np.nanmean(Amean_weight)

    save_path = os.path.join(smiles_viz_dir, f'{mol_name.replace("/", "")}.png')

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, Amean_weight-nanMean,
                                                     alpha=0.3,
                                                     size=(300, 300))

    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)