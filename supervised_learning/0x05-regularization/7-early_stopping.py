#!/usr/bin/env python3
"""
    Regularization project
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ Determines if gradient descent should be stopped early
        cost is current validation cost of the NN
        opt_cost is lowest recorded validation cost
        threshold is threshold used for early stopping
        patience is patience count used for early stopping
        count is count of how long threshold not been met
        Returns: boolean of whether NN should be stopped,
        followed by updated count
    """
    booly = False
    if cost >= opt_cost - threshold:
        count += 1
        if count == patience:
            return True, count
    else:
        booly = False
        count = 0
    return booly, count
