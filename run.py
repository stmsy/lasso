#!/usr/bin/env python

import argparse

import numpy as np

from dataload.load_data import load_data
from dataload.split_data import split_data
from learning.eval_model import get_score
from learning.load_model import load_model
from utils.yaml import load_yaml


def get_argument():
    """Get argument from standard input"""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1, type=str,
                        help="name of the config file in YAML format")
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    # Load config/settings from the YAML file
    arg = get_argument()
    filename = arg.filename[0]
    config = load_yaml(filename)

    # Define the model with the specified parameters
    model = load_model(config)

    # Load and standarilize the dataset
    dataset = load_data(config['dataset'])
    mean = np.mean(dataset.data, axis=0)
    std = np.std(dataset.data, axis=0)
    X = (dataset.data - mean) / std
    y = dataset.target
    np.set_printoptions(formatter={'float': '{: 0.10f}'.format})
    # Split the dataset for train and test
    train_params = config['train']['params']
    X_train, X_test, y_train, y_test = split_data(X, y, train_params)

    # Train the model with the train dataset
    model.fit(X_train, y_train)
    print("model.coefs:", model.coefs)
    y_pred = model.predict(X_test)
    # Get the test result
    score = get_score(train_params, y_test, y_pred)
    print("score:", score)
