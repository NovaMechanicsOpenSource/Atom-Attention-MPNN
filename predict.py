"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.args import PredictArgs
from chemprop.make_predictions import make_predictions


if __name__ == "__main__":
    make_predictions(args=PredictArgs().parse_args())
