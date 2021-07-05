#!/usr/bin/env python3
"""
    Training to play Atari's Breakout
"""
import rl
import keras as K
import gym
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')
        # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`.
        # In this case, however, we would need to store a `float32`
        # array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def create_q_model():
    """ q-learning model """
    # Network defined by the Deepmind paper
    inputs = K.layers.Input((actions,) + INPUT_SHAPE)
    perm = K.layers.Permute((2, 3, 1))(inputs)
    #  (84, 84, 4,)) #  # shape=(1, 1, 128)) # (84, 84, 4,))
    # Convolutions on the frames on the screen
    layer1 = K.layers.Conv2D(32, 8, name='conv1', strides=4,
                             activation="relu")(perm)
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
    memory = SequentialMemory(limit=1750000, window_length=actions)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.05,
                                  nb_steps=1750000)
    processor = AtariProcessor()
    agent = DQNAgent(model, policy=policy, test_policy=None,
                     processor=processor, enable_double_dqn=True,
                     enable_dueling_network=False,
                     dueling_type='avg', nb_actions=actions, memory=memory,
                     nb_steps_warmup=1750000, train_interval=4, delta_clip=1.)
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
    dqn.fit(env, nb_steps=1750000,
            visualize=False,
            verbose=2)  # ,
    # callbacks=callbacks)
    dqn.save_weights('policy.h5', overwrite=True)
