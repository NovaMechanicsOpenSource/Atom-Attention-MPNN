"""Pretraining a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.pretraining_mpn import pre_train

if __name__ == '__main__':
    pre_train(args=TrainArgs().parse_args())