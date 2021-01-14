#!/usr/bin/env python3
"""
    Optimization project
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Updates a variable using RMSProp opmtimizer
        (alpha, beta, eps, W, dW, dW_prev)

        alpha is learning rate
        beta2 is RMSProp weight
        epsilon is small number to avoid division by zero
        var is ndarray, var to update
        grad is ndarray containing gradient of var
        s is previous second moment of var

        Returns: updated var and new moment
    """
    Sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    W = var - alpha * (grad / (Sdw ** 0.5 + epsilon))
    return W, Sdw
