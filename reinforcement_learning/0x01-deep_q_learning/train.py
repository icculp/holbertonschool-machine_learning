#!/usr/bin/env python3
"""
    Training to play Atari's Breakout
"""
import rl
import keras as K
import gym
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


def create_q_model():
    """ q-learning model """
    # Network defined by the Deepmind paper
    inputs = K.layers.Input((actions,) + shp)
    #  (84, 84, 4,)) #  # shape=(1, 1, 128)) # (84, 84, 4,))
    # Convolutions on the frames on the screen
    layer1 = K.layers.Conv2D(32, 8, name='conv1', strides=4,
                             activation="relu")(inputs)
    layer2 = K.layers.Conv2D(64, 4, name='conv2', strides=2,
                             activation="relu")(layer1)
    layer3 = K.layers.Conv2D(64, 3, name='conv3', strides=1,
                             activation="relu")(layer2)
    layer4 = K.layers.Flatten()(layer3)
    layer5 = K.layers.Dense(512, name='dense1',
                            activation="relu")(layer4)
    action = K.layers.Dense(actions, name='dense2',
                            activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)


def build_agent(model, actions):
    """ build's the DQN agent """
    memory = SequentialMemory(limit=10000, window_length=actions)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05,
                                  nb_steps=10000)
    # processor = AtariProcessor()
    agent = DQNAgent(model, policy=policy, test_policy=None,
                     enable_double_dqn=True,
                     enable_dueling_network=False,
                     dueling_type='avg', nb_actions=actions, memory=memory,
                     nb_steps_warmup=10000, train_interval=4, delta_clip=1.)
    return agent


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    # 'Breakout-ram-v0')  # Breakout-v0')
    env.reset()
    # height, width, channels =
    # print(env.observation_space.shape)
    shp = env.observation_space.shape
    actions = env.action_space.n
    model = create_q_model()
    # model_target = create_q_model()
    dqn = build_agent(model, actions)
    dqn.compile(K.optimizers.Adam(lr=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=10000,
            visualize=False,
            verbose=2)  # ,
    # callbacks=callbacks)
    dqn.save_weights('policy.h5', overwrite=True)
