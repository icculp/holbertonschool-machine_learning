#!/usr/bin/env python3
"""
    Q-learning
"""
import numpy as np


def q_init(env):
    """ that initializes the Q-table:
        env is the FrozenLakeEnv instance
        Returns: the Q-table as a numpy.ndarray of zeros
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return Q
