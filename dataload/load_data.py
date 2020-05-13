#!/usr/bin/env python


def load_data(dataset):
    """Load the prepared/preprocessed dataset available from scikit-learn."""
    # TO-DO: The followint 'if' statement will be modified as necessary.
    if dataset == "boston":
        from sklearn.datasets import load_boston
        return load_boston()
