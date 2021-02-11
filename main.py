import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from abc import ABC, abstractmethod

# Actions: steer, gas, brake
# Observations: 96x96 rgb


class Agent(ABC):

    @abstractmethod
    def get_action(self, action_space: gym.Space):
        raise NotImplementedError

    def on_observation(self, observation, reward):
        pass


class RandomAgent(Agent):

    def get_action(self, action_space: gym.Space):
        return action_space.sample()


class KerasAgent(Agent):

    def __init__(self):
        super().__init__()

        # Create keras model
        self.model = tf.keras.Sequential([
            layers.Conv2D(512, (5, 5), input_shape=(84, 96, 1)),
            layers.Conv2D(256, (3, 3)),
            layers.Flatten(),
            layers.Dense(3, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def get_action(self, action_space: gym.Space):
        return [0, 1, 0]  # floor it!

    def on_observation(self, observation, reward):
        # Remove bottom of image because it just shows some stats
        observation = observation[:84, :]

        # Convert to grayscale
        observation = np.array(Image.fromarray(observation).convert('L'), dtype=np.float32)

        # plt.imshow(observation, cmap='gray')
        # plt.show()
        #
        # observation = np.expand_dims(observation, axis=2)
        #
        # x = np.array([observation])
        # y = np.array([[0.0, 1.0, 0.0]])
        #
        # self.model.fit(x, y)


def train(episodes: int):

    # Create environment
    # env = gym.make('CarRacing-v0').env # this one does not have a step limit
    env = gym.make('CarRacing-v0')

    # Create agent
    agent = KerasAgent()

    for episode in range(episodes):
        env.reset()

        total_reward = 0

        done = False
        step_index = 0
        while not done:

            # Show the environment
            env.render()

            # Chose action
            action = agent.get_action(env.action_space)

            # Perform action
            observation, reward, done, info = env.step(action)
            total_reward += reward

            # Check if we are done
            if done:
                print(f'Episode finished with reward: {total_reward}. Step: {step_index}')
                break

            agent.on_observation(observation, reward)

            step_index += 1

    # Close the environment
    env.close()


if __name__ == '__main__':
    train(10_000)
