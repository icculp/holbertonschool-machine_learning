#!/usr/bin/env python3
"""
    Q-learning
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ that uses epsilon-greedy to determine the next action
        Q is a numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation
        You should sample p with numpy.random.uniformn to determine
            if your algorithm should explore or exploit
        If exploring, you should pick the next action with
            numpy.random.randint from all possible actions
        Returns: the next action index
    """
    threshold = np.random.uniform(0, 1)
    if threshold > epsilon:
        i = np.argmax(Q[state, :])
    else:
        i = np.random.randint(Q.shape[1])
    return i
