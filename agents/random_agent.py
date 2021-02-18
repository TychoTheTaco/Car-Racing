import gym
from agents import Agent


class RandomAgent(Agent):

    def get_action(self, observation, action_space: gym.Space):
        return action_space.sample()
