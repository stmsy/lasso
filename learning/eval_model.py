#!/usr/bin/env python


def get_score(train_params, y_test, y_pred):
    """Evaluate the tested model with the specified metrics."""
    # TO-DO: The followint 'if' statement will be modified as necessary.
    if train_params['metrics'] == 'mse':
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_test, y_pred)
