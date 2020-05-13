#!/usr/bin/env python

import os

import yaml

from settings import CONFIG_DIR


YAML_EXTS = ('yaml', 'yml')


def is_filename_valid(filename, exts=YAML_EXTS):
    """Checks whether the filename ends with extension 'yaml' or 'yml'."""
    ext = filename.split('.')[-1]
    if ext in YAML_EXTS:
        return True
    else:
        return False


def load_yaml(filename):
    """Load several parameters in the YAML file as the Python dictionary.

    Arg:
      filename (str): name of the file in YAML format

    Return:
      params (dict): dictionary of parameters read from config file
    """
    if is_filename_valid(filename):
        filepath = os.path.join(CONFIG_DIR, filename)
        with open(filepath, 'r') as f:
            params = yaml.load(f)
        return params
    raise Exception("file is not valid")
