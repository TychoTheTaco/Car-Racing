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


def evaluate(env, agent, video_path=None, fps: int = 50, render=True, episodes: int = 1):
    """
    Evaluate an agent in an environment.
    :param env:
    :param agent:
    :param video_path:
    :param fps:
    :param render:
    :return:
    """
    video_writer = None

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

            # Render the environment
            frame = env.render(mode='rgb_array')
            if render:
                env.render()

            # Chose action
            action = agent.get_action(observation, env.action_space)

            # Save to video
            if video_writer is not None:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr_frame, f'S: {action[0]:.02f}', (540, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_frame, f'G: {action[1]:.02f}', (540, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_frame, f'B: {action[2]:.02f}', (540, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(bgr_frame)

            # Perform action
            new_observation, reward, done, info = env.step(action)
            episode_reward += reward

            observation = new_observation

        print(f'Episode {episode} | Score: {episode_reward:.02f}')

    if video_writer is not None:
        video_writer.release()
