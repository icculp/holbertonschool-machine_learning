#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ creates learning rate decay operation in tensorflow
        using inverse time decay
        learning_rate_decay(alpha_init, 1, i, 10)

        alpha is the original learning rate
        decay_rate is the weight dermining alpha decay
        global_step is number of gradient descent passes elapsed
        decay_step is number of passes to occur before alpha decayed again

        Returns: updated value for alpha
    """
    opt = tf.train.inverse_time_decay(alpha, global_step,
                                      decay_step, decay_rate,
                                      staircase=True)
    return opt
