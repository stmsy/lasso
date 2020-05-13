#!/usr/bin/env python


def load_model(config):
    """Load the lasso model as specified in the config file."""
    model_params = config['model']['params']
    # TO-DO: The followint 'if' statement will be modified as necessary.
    if config['model']['name'] == 'admm':
        from models.admm import ADMM
        return ADMM(lambda_=model_params['lambda'], rho=model_params['rho'],
                    max_iter=model_params['max_iter'])
