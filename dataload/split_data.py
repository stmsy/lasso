#!/usr/bin/env python

from sklearn.model_selection import train_test_split


def split_data(X, y, train_params):
    """Split the dataset for train and test with the specified settings."""
    return train_test_split(X, y, test_size=train_params['test_size'],
                            random_state=train_params['random_state'])
