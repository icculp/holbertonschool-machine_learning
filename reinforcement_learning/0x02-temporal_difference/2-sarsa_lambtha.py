#!/usr/bin/env python3
"""
    Temporal Difference
"""
import numpy as np


def asarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        performs the TD(λ) algorithm

        env is the openAI environment instance
        Q is a numpy.ndarray of shape (s,) containing the Q table
        lambtha is the elgibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the min value epsilon should decay to
        epsilon_decay is decay rate for updating epsilon between episodes
        Returns: Q, the updated value estimate
    """
    # print(V)
    states = Q.shape[0]
    elig = np.zeros(Q.shape)
    # se = set()
    for i in range(episodes):
        s = env.reset()
        action = np.argmax(Q[s])
        # break
        # episode = []
        # elig = np.zeros(states)
        for j in range(max_steps):
            # action = policy(s)
            s_new, reward, done, info = env.step(action)
            action_new = np.argmax(Q[s_new])
            # episode.append([s, action, reward, s_new])
            # elig[s] *= lambtha * gamma
            #elig += 1.0
            # elig[s, :] *= 0
            elig *= gamma * epsilon
            elig[s, action] += (1.0)  # - epsilon)
            
            # se.add(s)
            delta = reward + gamma * Q[s_new, action_new] - Q[s, action]
            Q += alpha * delta * elig  # * epsilon
            # for si in range(len(V)):
            # print(V.shape, elig.shape)
            if done:
                break  # s = env.reset()
            else:
                s = s_new
                action = action_new

        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon -= epsilon_decay 
    # print(se)
    return Q


'''        s = s_new
episode = np.array(episode, dtype=int)
G = 0
for j, step in enumerate(episode[::-1]):
    s, action, reward, s_next = step
    G = gamma * G + reward
    #if s not in episode[:i, 0]:
    V[s] = V[s] + alpha * (G - V[s])'''



def epsilon_greedy(env, Q, state, epsilon):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env,
                  Q,
                  lambtha,
                  episodes=5000,
                  max_steps=100,
                  alpha=0.1,
                  gamma=0.99,
                  epsilon=1,
                  min_epsilon=0.1,
                  epsilon_decay=0.05):
    init_epsilon = epsilon

    Et = np.zeros((Q.shape))

    for i in range(episodes):
        state = env.reset()

        action = epsilon_greedy(env, Q, state, epsilon)

        for j in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1.0

            new_state, reward, done, info = env.step(action)
            new_action = epsilon_greedy(env, Q, state, epsilon)

            delta_t = reward + gamma + Q[new_state, new_action] - Q[state,
                                                                    action]

            Q[state, action] = Q[state, action] + alpha * delta_t * Et[state,
                                                                       action]

            if done:
                break
            state = new_state
            action = new_action

        epsilon = (min_epsilon + (init_epsilon - min_epsilon)
                   * np.exp(- epsilon_decay * i))

    return Q