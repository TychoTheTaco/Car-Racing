from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from agents import *
import random
import time
from pathlib import Path
import datetime
from PIL import Image


class DeepQNetworkAgent(Agent):

    def __init__(self, env: gym.Env, model_path=None):
        self._env = env

        self._learning_rate = 0.001
        self._epsilon = 1
        self._min_epsilon = 0.01
        self._epsilon_decay = 0.995
        self._discount = 0.99
        self._batch_size = 512

        if model_path is None:
            self._model = tf.keras.Sequential([
                layers.Input(shape=(28, 32, 1)),
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(10, activation='linear')
            ])
            self._model.compile(optimizer=tf.optimizers.Adam(learning_rate=self._learning_rate), loss='mse')
        else:
            self._model = tf.keras.models.load_model(model_path)
        self._model.summary()

        self._previous_observation = None

    def get_action(self, observation, action_space: gym.Space):
        p = self._model.predict(np.array([self._get_observation(observation)]))
        discrete_action = np.argmax(p[0])
        return self._get_continuous_action(discrete_action)

    def _get_continuous_action(self, discrete_action):
        return (
            (-1, 0.01, 0),
            (-1, 0, 0.01),
            (-0.5, 0.01, 0),
            (-0.5, 0, 0.01),

            (0, 0.01, 0),
            (0, 0, 0.01),

            (0.5, 0.01, 0),
            (0.5, 0, 0.01),
            (1, 0.01, 0),
            (1, 0, 0.01),
        )[discrete_action]

    def _get_observation(self, observation):

        # Remove bottom of image because it just shows some stats, only keep green channel
        observation = observation[:84, :, 1]

        # Reduce resolution
        observation = observation[::3, ::3]

        observation = np.array(observation, dtype=np.float32)

        observation = np.expand_dims(observation, axis=2)

        return observation

    def train(self, episodes: int = 1000):
        start_time = time.time()

        model_dir = Path('models', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

        # Episode interval to save the model
        save_interval = 50

        # Keep track of reward earned each episode
        episode_rewards = []

        replay_buffer = deque(maxlen=500_000)

        for episode in range(1, episodes + 1):

            # Reset environment
            observation = self._env.reset()
            observation = self._get_observation(observation)
            episode_reward = 0

            # Simulate steps
            done = False
            step = 0
            while not done:

                # Chose action
                if np.random.rand() < self._epsilon:
                    discrete_action = np.random.randint(0, 10)
                else:
                    p = self._model.predict(np.array([observation]))
                    discrete_action = np.argmax(p[0])

                # Perform action
                new_observation, reward, done, info = self._env.step(self._get_continuous_action(discrete_action))
                new_observation = self._get_observation(new_observation)
                episode_reward += reward

                #if step >= 250:
                #    import matplotlib.pyplot as plt
                #    plt.imshow(new_observation, cmap='gray')
                #    plt.show()

                # End early if our score is bad
                if episode_reward <= -15:
                    reward = -100
                    done = True

                # Save transition to replay buffer
                transition = (observation, discrete_action, reward, done, new_observation)
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
            print(f'Episode {episode} Reward: {int(episode_reward)} AVG: {np.average(episode_rewards):.04f} EPSILON: {self._epsilon:.04f} BUFFER: {len(replay_buffer)} STEPS: {step}')

            # Save model
            if not episode % save_interval:
                print('Saving model...')
                model_dir.mkdir(parents=True, exist_ok=True)
                self._model.save(model_dir / f'episode-{episode}.h5')

        # Close the environment
        self._env.close()

        print(f'Finished training in {(time.time() - start_time) / 60} minutes')