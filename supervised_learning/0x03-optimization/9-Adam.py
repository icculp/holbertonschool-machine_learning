#!/usr/bin/env python3
"""
    Optimization project
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """ Updates a variable using Adam opmtimizer
        (0.001, 0.9, 0.99, 1e-8,
        W, dW, dW_prev1, dW_prev2, i + 1)

        alpha is the learning rate
        beta1 is RMSProp weight
        beta2 is RMSProp weight
        epsilon is small number to avoid division by zero
        var is ndarray and var to be updated
        grad contains gradient of var
        v is previous first moment of var
        s is previous second moment
        t is time step used for bias correction

        Returns: updated var, new moment1, new moment2
    """
    moment1 = beta1 * v + (1 - beta1) * grad
    moment2 = beta2 * s + (1 - beta2) * (grad ** 2)
    moment1E = moment1 / (1 - (beta1 ** t))
    moment2E = moment2 / (1 - (beta2 ** t))
    w = var - (alpha * (moment1E / ((moment2E ** 0.5) + epsilon)))
    return w, moment1, moment2
