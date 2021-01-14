#!/usr/bin/env python3
"""
    Optimization project
"""


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ updates the learning rate in stepwise fasion
        using inverse time decay in numpy
        learning_rate_decay(alpha_init, 1, i, 10)

        alpha is the original learning rate
        decay_rate is the weight dermining alpha decay
        global_step is number of gradient descent passes elapsed
        decay_step is number of passes to occur before alpha decayed again

        Returns: updated value for alpha
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
