from abc import ABC, abstractmethod
import gym


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation, action_space: gym.Space):
        """
        Get a valid action from the action space.
        :param observation: The current observation of the environment.
        :param action_space: The action space of the environment
        :return: A valid action from the action space that can be passed directly to env.step()
        """
        raise NotImplementedError


class RandomAgent(Agent):

    def get_action(self, observation, action_space: gym.Space):
        return action_space.sample()
