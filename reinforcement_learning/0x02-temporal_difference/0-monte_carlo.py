#!/usr/bin/env python3
"""
    Temporal Difference
"""
import numpy as np


def run_game(env, policy, display=True):
    """ run game for training """
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s
        if display:
            clear_output(True)
            env.render()
            sleep(1)

        timestep = []
        timestep.append(s)
        n = np.random.uniform(0, sum(policy[s].values()))
        top_range = 0
        for prob in policy[s].items():
                top_range += prob[1]
                if n < top_range:
                    action = prob[0]
                    break 
        state, reward, finished, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    if display:
        clear_output(True)
        env.render()
        sleep(1)
    return episode


def create_state_action_dictionary(env, policy):
    """ state action dict """
    Q = {}
    for key in policy.keys():
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def create_random_policy(env):
    """ policy bullshit """
    policy = {}
    for key in range(0, env.observation_space.n):
        current_end = 0
        p = {}
        for action in range(0, env.action_space.n):
            p[action] = 1 / env.action_space.n
        policy[key] = p
    return policy


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
        performs the Monte Carlo algorithm:

        env is the openAI environment instance
        V is a numpy.ndarray of shape (s,) containing the value estimate
        policy is a function that takes in a state and
            returns the next action to take
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        Returns: V, the updated value estimate
    """
    epsilon = 1 - gamma
    #print(V.shape, V)
    #print(env.observation_space.n)
    #print('policy', policy())
    returns = {}
    Policy = policy
    policy = create_random_policy(env)
    print(policy)
    Q = create_state_action_dictionary(env, policy)
    print(Q)
    for ep in range(episodes):
        G = 0
        episode = run_game(env=env, policy=policy, display=False)

        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            state_action = (s_t, a_t)
            G += r_t

            if not state_action in [(x[0], x[1]) for x in episode[0:i]]: # 
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]   
                    
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes
                
                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = np.random.choice(indices)
                
                A_star = max_Q # 14.
                
                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))
    print(policy)

    return policy
    return V