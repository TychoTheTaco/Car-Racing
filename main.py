import logging
import matplotlib.pyplot as plt

import gym
import minerl

logging.basicConfig(level=logging.DEBUG)


def random_agent():
    env = gym.make('MineRLNavigateExtremeVectorObf-v0')

    obs = env.reset()
    print(obs)

    done = False

    x = []
    angles = []

    i = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        x.append(i)
        angles.append(obs['compassAngle'])

        i += 1

    plt.plot(x, angles)
    plt.show()


def good_agent():
    env = gym.make('MineRLNavigateExtremeVectorObf-v0')

    obs = env.reset()
    print(obs)

    done = False
    net_reward = 0

    x = []
    angles = []

    i = 0

    while not done:
        action = env.action_space.noop()

        action['camera'] = [0, 0.03 * obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Total reward: ", net_reward)

        x.append(i)
        angles.append(obs['compassAngle'])

        i += 1

    plt.plot(x, angles)
    plt.show()


if __name__ == '__main__':
    good_agent()
