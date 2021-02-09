import logging

import gym
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Actions: steer, gas, brake
# Observations: 96x96 rgb

if __name__ == '__main__':

    #env = gym.make('CarRacing-v0').env
    env = gym.make('CarRacing-v0')
    env.reset()

    total = 0

    for step in range(1000):

        # Show the environment
        env.render()

        # Chose action
        action = env.action_space.sample()

        # Perform action
        observation, reward, done, info = env.step(action)

        # Remove bottom of image
        observation = observation[:84, :]

        total += reward
        print(total)

        if done:
            print("Episode finished after {} timesteps".format(step))
            print(info)
            break

        env.step(env.action_space.sample())
    env.close()
