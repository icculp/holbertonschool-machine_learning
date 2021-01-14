#!/usr/bin/env python3
"""
    Optimization project
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates tf variable using gradient descent w/ momentum
        alpha is learning rate
        beta1 is momentum weight
        var is ndarray contaning var to be updated
        grad contains gradient of var
        v is previous first moment of var
        Returns: updated variable and new moment
    """
    Vdw = beta1 * v + ((1 - beta1) * grad)
    W = var - (alpha * Vdw)
    return W, Vdw
