from collections import deque
from car_racing_environment import CarRacing
import numpy as np


class CustomCarRacing(CarRacing):
    """
    This is a wrapper around the standard 'CarRacing' environment. It modifies the behavior of the step(), including changes to the observation space and changes to the reward
    amount. This environment wrapper is intended to be used with the PPO agent.
    """

    def __init__(self, image_stack_size=4, step_repeat=8):
        """
        Create a new environment.
        :param image_stack_size: The number of frames to "stack" together into a single observation. This is useful for getting velocity information.
        :param step_repeat: The number of times to repeat internal calls to step() for every external call. This greatly speeds up training since each frame is very similar. From
        the outside it will appear as if the environment is running at FPS / STEP_REPEAT frames per second, where FPS is the original FPS of the environment.
        """
        super().__init__(verbose=0)
        self._image_stack_size = image_stack_size
        self._step_repeat = step_repeat

        # A deque is used to limit the size of the image stack.
        self._image_stack = deque(maxlen=self._image_stack_size)

        # Keep track of past rewards. This is used to end the simulation early if the agent consistently performs poorly
        self._reward_history = deque([0] * 100, maxlen=100)

    def reset(self):
        self._image_stack.clear()
        self._reward_history.extend([0] * 100)
        return super().reset()

    def step(self, action):
        total_reward = 0

        # Repeat steps, accumulating the rewards
        for i in range(self._step_repeat):
            observation, reward, done, info = super().step(action)

            # Punish the agent for going off of the track
            if np.mean(observation[64:80, 42:54, 1]) > 120:
                reward -= 0.05

            # End early if the agent consistently does poorly
            self._reward_history.append(reward)
            if np.mean(self._reward_history) < -0.1:
                done = True

            total_reward += reward
            if done:
                break

        # Add the latest observation to the image stack
        observation = self._preprocess_observation(observation)
        self._image_stack.append(observation)
        while len(self._image_stack) < self._image_stack_size:
            self._image_stack.append(observation)

        # Convert image stack to numpy array
        image_stack_array = np.empty((32, 32, self._image_stack_size))
        for i in range(self._image_stack_size):
            image_stack_array[..., i] = self._image_stack[i]

        return image_stack_array, total_reward, done, info

    @staticmethod
    def _preprocess_observation(observation):

        # Keep only the green channel of the RGB image and reduce the resolution
        observation = observation[::3, ::3, 1]

        # Convert to float
        observation = observation.astype(np.float32)

        # Normalize values between -1 and 1
        observation = (observation / 128) - 1

        return observation
