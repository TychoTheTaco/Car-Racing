from abc import ABC, abstractmethod
import gym


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation, action_space: gym.Space):
        raise NotImplementedError

    def on_observation(self, observation, reward: float, done: bool):
        pass


class RandomAgent(Agent):

    def get_action(self, observation, action_space: gym.Space):
        return action_space.sample()
