from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from abc import ABC, abstractmethod

# Actions: steer, gas, brake
# Observations: 96x96 rgb

# Reward: -0.1 per frame, +1000/N per tile (N == tiles visited so far)

DISCOUNT = 0.95


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation, action_space: gym.Space):
        raise NotImplementedError

    def on_observation(self, observation, reward: float, done: bool):
        pass


class RandomAgent(Agent):

    def get_action(self, observation, action_space: gym.Space):
        return action_space.sample()


BUCKET_COUNT = 10
MAX_OBSERVATION_HISTORY = 1000
class KerasAgent(Agent):

    def __init__(self):
        super().__init__()

        # Output continuous values are converted to discrete values determined by BUCKET_COUNT

        # Create keras model
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), input_shape=(84, 96, 1), activation='relu'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(3 * BUCKET_COUNT, activation='linear'),
            layers.Reshape((3, BUCKET_COUNT))
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        self._previous_observation = None

        self._observation_history = deque(maxlen=MAX_OBSERVATION_HISTORY)

    def get_action(self, observation, action_space: gym.Space):
        observation = self._reformat_observation(observation)
        future_q_values = self.model.predict(np.array([observation]))[0]
        action = np.interp(np.argmax(future_q_values, axis=1), np.arange(0, 10), np.linspace(-1, 1, BUCKET_COUNT))
        print(action)
        return action

    def on_observation(self, observation, reward: float, done: bool):
        observation = self._reformat_observation(observation)

        if self._previous_observation is not None:
            future_q = self.model.predict(np.array([observation]))[0]

            if done:
                new_q = reward
            else:
                max_future_q = np.max(future_q, axis=1)
                new_q = reward + DISCOUNT * max_future_q

            x = np.array([self._previous_observation])
            y = np.full((1, 3, BUCKET_COUNT), -1)

            am = np.argmax(future_q, axis=1)
            for i, v in enumerate(am):
                y[0, i, v] = new_q[i]

            self.model.fit(x, y)

        self._previous_observation = observation

    def _reformat_observation(self, observation):
        # Remove bottom of image because it just shows some stats
        observation = observation[:84, :]

        # Convert to grayscale
        observation = np.array(Image.fromarray(observation).convert('L'), dtype=np.float32)

        observation = np.expand_dims(observation, axis=2)

        return observation


def train(episodes: int):

    # Create environment
    # env = gym.make('CarRacing-v0').env # this one does not have a step limit
    env = gym.make('CarRacing-v0')

    # Create agent
    agent = KerasAgent()

    for episode in range(episodes):
        print(f'Episode {episode}')
        observation = env.reset()

        total_reward = 0

        previous_observation = None

        done = False
        step_index = 0
        while not done:

            # Show the environment
            env.render()

            # Chose action
            action = agent.get_action(observation, env.action_space)

            # Perform action
            observation, reward, done, info = env.step(action)
            total_reward += reward

            # Notify agent of observation
            agent.on_observation(observation, reward, done)

            # Check if we are done
            if done:
                print(f'Episode finished with reward: {total_reward}. Step: {step_index}')
                break

            if previous_observation is not None and np.allclose(previous_observation, observation):
                print('STOP')
                break

            if total_reward < -20:
                print('STOP')
                break

            step_index += 1
            previous_observation = observation
        print(f'Ended after {step_index} steps.')

    # Close the environment
    env.close()


if __name__ == '__main__':
    train(10_000)
