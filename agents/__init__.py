from abc import ABC, abstractmethod
import gym
import cv2


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


def evaluate(env, agent, video_path=None, fps: int = 50):
    """
    Evaluate an agent in an environment.
    :param env:
    :param agent:
    :param video_path:
    :param fps:
    :return:
    """

    # Reset environment
    observation = env.reset()
    episode_reward = 0

    # Create video writer
    video_writer = None
    if video_path is not None:
        frame = env.render(mode='rgb_array')
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

    # Simulate steps
    done = False
    while not done:

        # Render the environment
        frame = env.render(mode='rgb_array')

        # Save to video
        if video_writer is not None:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Chose action
        action = agent.get_action(observation, env.action_space)

        # Perform action
        new_observation, reward, done, info = env.step(action)
        episode_reward += reward

    if video_writer is not None:
        video_writer.release()
