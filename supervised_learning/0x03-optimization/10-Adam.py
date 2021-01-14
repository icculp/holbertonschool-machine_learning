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

        Returns: RMSProp optimiization operation
    """
    opt = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return opt.minimize(loss)
