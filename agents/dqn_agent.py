import gym
from agents import Agent
from collections import deque

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
import random
import time
from pathlib import Path
import datetime


class DQNAgent(Agent):

    def __init__(self, env: gym.Env, model_path=None, epsilon: float = 1.0):
        self._env = env

        self._learning_rate = 0.001
        self._epsilon = epsilon
        self._min_epsilon = 0.01
        self._epsilon_decay = 0.995
        self._discount = 0.99
        self._batch_size = 64

        if model_path is None:
            self._model = tf.keras.Sequential([
                layers.Input(shape=env.observation_space.shape),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(4, activation='linear')
            ])
            self._model.compile(optimizer=tf.optimizers.Adam(learning_rate=self._learning_rate), loss='mse')
        else:
            self._model = tf.keras.models.load_model(model_path)
        self._model.summary()

        self._previous_observation = None

    def get_action(self, observation, action_space: gym.Space):
        p = self._model.predict(np.array([observation]))
        return np.argmax(p[0])

    def train(self, episodes: int = 1000):
        start_time = time.time()

        model_dir = Path('models', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        model_dir.mkdir(parents=True, exist_ok=True)

        # Episode interval to save the model
        save_interval = 50

        # Keep track of reward earned each episode
        episode_rewards = []

        replay_buffer = deque(maxlen=500_000)

        for episode in range(1, episodes + 1):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0

            # Simulate steps
            done = False
            step = 0
            while not done:

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
                    dones = np.array([x[3] for x in samples])
                    new_states = np.array([x[4] for x in samples])

                    # Get estimated Q values for initial states
                    current_q_values = self._model.predict_on_batch(states)

                    # Calculate new Q values
                    max_q_values = np.amax(self._model.predict_on_batch(new_states), axis=1)
                    new_q_values = rewards + self._discount * max_q_values * (1 - dones)

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

            # Save model
            if not episode % save_interval:
                print('Saving model...')
                self._model.save(model_dir / f'episode-{episode}.h5')

        # Close the environment
        self._env.close()

        print(f'Finished training in {(time.time() - start_time) / 60} minutes')
