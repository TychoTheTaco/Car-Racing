from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from abc import ABC, abstractmethod
from agents.agents import *


class QLearningAgent(Agent):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self._q_table = np.zeros(())

    def get_action(self, observation, action_space: gym.Space):
        pass

    def on_observation(self, observation, reward: float, done: bool):
        super().on_observation(observation, reward, done)


class DeepQLearningAgent(Agent):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self._epsilon = 1
        self._min_epsilon = 0.01

        self._discount = 0.95

        self._model = tf.keras.Sequential([
            layers.Input(shape=observation_space.shape),
            layers.Dense(64),
            layers.Dense(32),
            layers.Dense(4)
        ])
        self._model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mse')
        print(observation_space.shape)
        self._action_index = -1

        self._previous_observation = None

    def get_action(self, observation, action_space: gym.Space):
        self._action_index += 1

        if np.random.random() < self._get_epsilon(self._action_index):
            return action_space.sample()

        p = self._model.predict(np.array([observation]))
        return np.argmax(p[0])

    def on_observation(self, observation, reward: float, done: bool, action):

        if self._previous_observation is not None:
            current_q = self._model.predict(np.array([self._previous_observation]))[0]
            future_q = self._model.predict(np.array([observation]))[0]

            if done:
                new_q = reward
            else:
                max_future_q = np.max(future_q)
                new_q = reward + self._discount * max_future_q

            current_q[action] = new_q

            x = np.array([self._previous_observation])
            y = np.array([current_q])

            self._model.fit(x, y, verbose=0)

        self._previous_observation = observation

    def _get_epsilon(self, t):
        return max(100 / (t + 1), self._min_epsilon)


def train(episodes: int):

    # Create environment
    env = gym.make('LunarLander-v2')

    # Create agent
    agent = DeepQLearningAgent(env.observation_space, env.action_space)

    episode_rewards = []
    for episode in range(episodes):
        print(f'Episode {episode}')

        # Reset environment
        observation = env.reset()
        agent.on_observation(observation, 0, False, 0)

        episode_reward = 0

        done = False
        step_index = 0
        while not done:

            # Show the environment
            #env.render()

            # Chose action
            action = agent.get_action(observation, env.action_space)

            # Perform action
            observation, reward, done, info = env.step(action)
            episode_reward += reward

            # Notify agent of observation
            agent.on_observation(observation, reward, done, action)

            # Check if we are done
            if done:
                break

            step_index += 1

        episode_rewards.append(episode_reward)
        print(f'Ended after {step_index} steps. Reward: {int(episode_reward)} AVG: {np.average(episode_rewards)}')

    # Close the environment
    env.close()


if __name__ == '__main__':
    train(1000)
