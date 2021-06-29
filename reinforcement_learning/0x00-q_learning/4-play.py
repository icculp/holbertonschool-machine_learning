#!/usr/bin/env python3
"""
    Q-learning
"""
from functools import total_ordering
import numpy as np


def play(env, Q, max_steps=100):
    """ that has the trained agent play an episode

        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
        Each state of the board should be displayed via the console
        You should always exploit the Q-table
        Returns: the total rewards for the episode
    """
    state = env.reset()
    total_rewards = 0
    env.render()
    for step in range(max_steps):
        # clear_output(wait=True)
        action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        state = new_state
        total_rewards += reward
        env.render()
        if done:
            break
    return total_rewards
