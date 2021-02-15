from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from abc import ABC, abstractmethod
from agents.agents import *
import random


class QLearningAgent(Agent):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        self._q_table = np.zeros(())

    def get_action(self, observation, action_space: gym.Space):
        pass

    def on_observation(self, observation, reward: float, done: bool):
        super().on_observation(observation, reward, done)


class DeepQNetworkAgent:

    def __init__(self, env: gym.Env):
        self._env = env

        self._learning_rate = 0.001
        self._epsilon = 1
        self._min_epsilon = 0.01
        self._epsilon_decay = 0.995
        self._discount = 0.99
        self._batch_size = 64

        self._model = tf.keras.Sequential([
            layers.Input(shape=env.observation_space.shape),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(4, activation='linear')
        ])
        self._model.compile(optimizer=tf.optimizers.Adam(learning_rate=self._learning_rate), loss='mse')
        self._model.summary()

        self._previous_observation = None

    # def on_observation(self, observation, reward: float, done: bool, action):
    #
    #     if self._previous_observation is not None:
    #         current_q = self._model.predict(np.array([self._previous_observation]))[0]
    #         future_q = self._model.predict(np.array([observation]))[0]
    #
    #         if done:
    #             new_q = reward
    #         else:
    #             max_future_q = np.max(future_q)
    #             new_q = reward + self._discount * max_future_q
    #
    #         current_q[action] = new_q
    #
    #         x = np.array([self._previous_observation])
    #         y = np.array([current_q])
    #
    #         self._model.fit(x, y, verbose=0)
    #
    #     self._previous_observation = observation

    # def _get_epsilon(self, t):
    #     return max(100 / (t + 1), self._min_epsilon)

    def train(self, episodes: int = 1000):

        # Keep track of reward earned each episode
        episode_rewards = []

        replay_buffer = deque(maxlen=500_000)

        for episode in range(episodes):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0

            # Simulate steps
            done = False
            step = 0
            while not done:

                # Show the environment
                # env.render()

                # Chose action
                if np.random.rand() < self._epsilon:
                    action = self._env.action_space.sample()
                else:
                    p = self._model.predict(np.array([observation]))
                    action = np.argmax(p[0])

                # Perform action
                new_observation, reward, done, info = self._env.step(action)
                episode_reward += reward

                # Save transition to replay buffer
                transition = (observation, action, reward, done, new_observation)
                replay_buffer.append(transition)

                # Fit model
                if len(replay_buffer) >= self._batch_size and (step + 1) % 5 == 0:

                    # Get a random sample from the replay buffer
                    samples = random.sample(replay_buffer, self._batch_size)

                    states = np.array([x[0] for x in samples])
                    actions = [x[1] for x in samples]
                    rewards = [x[2] for x in samples]
                    dones = [x[3] for x in samples]
                    new_states = np.array([x[4] for x in samples])

                    # Get estimated Q values for initial states
                    current_q_values = self._model.predict_on_batch(states)

                    # Calculate new Q values
                    max_q_values = np.amax(self._model.predict_on_batch(new_states), axis=1)
                    new_q_values = rewards + self._discount * max_q_values * (1 - np.array(dones))

                    # Update target Q values
                    current_q_values[[np.arange(0, self._batch_size)], [actions]] = new_q_values

                    # Train model
                    self._model.fit(states, current_q_values, verbose=0)

                step += 1
                observation = new_observation

            # Decay epsilon
            self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)

            episode_rewards.append(episode_reward)
            print(f'Episode {episode} Reward: {int(episode_reward)} AVG: {np.average(episode_rewards)} EPSILON: {self._epsilon}')

        # Close the environment
        self._env.close()


def main():
    # Create environment
    # States: 8 [position x, position y, velocity x, velocity y, angle, angular velocity, left leg on ground, right leg on ground]
    # Actions: 4 [nothing, left engine, main engine, right engine]
    env = gym.make('LunarLander-v2')

    env.seed(21)
    np.random.seed(21)

    # Create agent
    agent = DeepQNetworkAgent(env)

    # Train agent
    agent.train(1000)


if __name__ == '__main__':
    main()
