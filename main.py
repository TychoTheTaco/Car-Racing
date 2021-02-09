import logging

import gym

logging.basicConfig(level=logging.DEBUG)

# Actions: steer, gas, brake
# Observations: 96x96 rgb

if __name__ == '__main__':

    env = gym.make('CarRacing-v0').env
    env.reset()

    total = 0

    for step in range(1000):

        # Show the environment
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total += reward
        print(total)

        if done:
            print("Episode finished after {} timesteps".format(step))
            print(info)
            break

        env.step(env.action_space.sample())
    env.close()
