"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.cross_validate import cross_validate
from chemprop.run_training import run_training


if __name__ == '__main__':
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
