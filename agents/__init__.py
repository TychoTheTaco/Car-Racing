from abc import ABC, abstractmethod
import gym
import cv2
import numpy as np
from typing import Any


class Agent(ABC):

    @abstractmethod
    def get_action(self, observation: Any, action_space: gym.Space) -> Any:
        """
        Get a valid action from the action space.
        :param observation: The current observation of the environment.
        :param action_space: The action space of the environment
        :return: A valid action from the action space that can be passed directly to env.step()
        """
        raise NotImplementedError


def evaluate(env, agent, video_path=None, fps: int = 50, render=True, episodes: int = 1, render_observation_video: bool = False):
    """
    Evaluate an agent in an environment.
    :param env:
    :param agent:
    :param video_path:
    :param fps:
    :param render:
    :param episodes:
    :param render_observation_video:
    :return:
    """
    video_writer = None
    observation_video_writer = None

    # Create observation video writer
    combined_observation = np.zeros((32 * 2, 32 * 2), dtype=np.uint8)
    if render_observation_video and video_path is not None:
        observation_video_writer = cv2.VideoWriter('observation-' + video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (combined_observation.shape[1], combined_observation.shape[0]))

    for episode in range(episodes):

        # Reset environment
        observation = env.reset()
        episode_reward = 0

        # Create video writer
        if video_writer is None and video_path is not None:
            frame = env.render(mode='rgb_array')
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

        # Simulate steps
        done = False
        while not done:

            # Save observation to video
            if observation_video_writer is not None:
                combined_observation[:32, :32] = np.interp(observation[..., 0], [-1, 1], [0, 1]) * 255
                combined_observation[:32, 32:] = np.interp(observation[..., 1], [-1, 1], [0, 1]) * 255
                combined_observation[32:, :32] = np.interp(observation[..., 2], [-1, 1], [0, 1]) * 255
                combined_observation[32:, 32:] = np.interp(observation[..., 3], [-1, 1], [0, 1]) * 255
                observation_video_writer.write(cv2.cvtColor(combined_observation, cv2.COLOR_GRAY2BGR))

            # Chose action
            action = agent.get_action(observation, env.action_space)

            # Perform action
            new_observation, reward, done, info = env.step(action)
            episode_reward += reward

            # Render the environment
            frame = env.render(mode='rgb_array')
            if render:
                env.render()

            # Save to video
            if video_writer is not None:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_frame, f'S: {action[0]:.02f}', (540, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_frame, f'G: {action[1]:.02f}', (540, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_frame, f'B: {action[2]:.02f}', (540, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(bgr_frame)

            observation = new_observation

        print(f'Episode {episode} | Score: {episode_reward:.02f}')

    if video_writer is not None:
        video_writer.release()
    if observation_video_writer is not None:
        observation_video_writer.release()
