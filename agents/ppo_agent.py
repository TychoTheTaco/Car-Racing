import datetime
from collections import deque
import tensorflow as tf
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow.keras import layers
import gym
from agents import Agent
import numpy as np
from pathlib import Path
from typing import Union, Optional
from car_racing_environment import CarRacing


try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.random.set_seed(0)


class PPOAgent(Agent):

    def __init__(self, env: gym.Env, model_path: Optional[Union[str, Path]] = None):
        self._env = env

        # Load model
        if model_path is not None:
            model_path = Path(model_path)
            self._model = tf.keras.models.load_model(model_path)
        else:
            self._model = self._create_model()
        self._model.summary()

    def get_action(self, observation, action_space: gym.Space):
        observation = np.expand_dims(observation, axis=0)
        alpha_beta = self._model.predict(observation)[0][0]
        alpha, beta = alpha_beta[:, 0], alpha_beta[:, 1]
        distribution = Beta(alpha, beta)
        action = distribution.sample().numpy()
        action[0] = np.interp(action[0], [0, 1], [-1, 1])
        return action

    def train(self, episodes: int = 1000, log_interval: int = 10, model_dir: str = 'models', save_interval: int = 100):
        model_dir = Path(model_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

        # Keep track of some stats
        episode_rewards = []

        BUFFER_SIZE = 2000
        BATCH_SIZE = 128
        GAMMA = 0.99

        PPO_EPOCH = 10
        CLIP_PARAM = 0.1

        transitions = []

        for episode in range(episodes):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0
            reward_history = []

            done = False
            while not done:

                # Choose action
                p = self._model(np.expand_dims(observation, axis=0))[0][0]
                alpha, beta = p[:, 0], p[:, 1]
                beta_distribution = Beta(alpha, beta)
                action = beta_distribution.sample()
                lp = beta_distribution.log_prob(action)
                lp = tf.reduce_sum(lp)

                A = action.numpy()
                A[0] = np.interp(A[0], [0., 1.], [-1., 1.])

                # Perform action
                new_observation, reward, done, _ = self._env.step(A)
                episode_reward += reward
                reward_history.append(reward)

                transitions.append((observation, action, lp, reward, new_observation))
                if len(transitions) == BUFFER_SIZE:
                    print('learning!')

                    states = tf.convert_to_tensor([x[0] for x in transitions])
                    actions = tf.convert_to_tensor([x[1] for x in transitions])
                    old_a_logp = tf.convert_to_tensor([x[2] for x in transitions])
                    rewards = tf.convert_to_tensor([x[3] for x in transitions])
                    new_states = tf.convert_to_tensor([x[4] for x in transitions])

                    old_a_logp = tf.expand_dims(old_a_logp, axis=1)
                    rewards = tf.expand_dims(rewards, axis=1)

                    target_v = rewards + GAMMA * self._model(new_states)[1]
                    adv = target_v - self._model(states)[1]

                    def gen_batches(indices, batch_size):
                        for i in range(0, len(indices), batch_size):
                            yield indices[i:i + batch_size]

                    for _ in range(PPO_EPOCH):
                        indices = np.arange(BUFFER_SIZE)
                        np.random.shuffle(indices)

                        for batch in gen_batches(indices, BATCH_SIZE):

                            with tf.GradientTape() as tape:
                                ab = self._model(tf.gather(states, batch))[0]
                                alpha, beta = ab[:, :, 0], ab[:, :, 1]
                                dist = Beta(alpha, beta)
                                a_logp = dist.log_prob(tf.gather(actions, batch))
                                a_logp = tf.reduce_sum(a_logp, axis=1, keepdims=True)
                                ratio = tf.exp(a_logp - tf.gather(old_a_logp, batch))
                                surr1 = ratio * tf.gather(adv, batch)
                                surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * tf.gather(adv, batch)
                                action_loss = -tf.minimum(surr1, surr2)
                                action_loss = tf.reduce_mean(action_loss)

                                one = self._model(tf.gather(states, batch))[1]
                                two = tf.gather(target_v, batch)
                                value_loss = tf.losses.mse(two, one)
                                value_loss = tf.reduce_mean(value_loss)

                                loss = action_loss + 2 * value_loss

                            g = tape.gradient(loss, self._model.trainable_variables)
                            self._model.optimizer.apply_gradients(zip(g, self._model.trainable_variables))

                    transitions.clear()

                observation = new_observation

            episode_rewards.append(episode_reward)

            # Print some statistics
            if not episode % log_interval:
                print(f'Episode {episode} | Reward: {episode_reward:.02f} | Average: {np.average(episode_rewards[-50:]):.02f}')

            # Save model
            if not episode % save_interval:
                model_dir.mkdir(parents=True, exist_ok=True)
                self._model.save(model_dir / f'episode-{episode}.h5')

    def _create_model(self):

        def create_conv_layer(filters, kernel_size, strides):
            return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                 bias_initializer=tf.initializers.constant(0.1))

        # Input is a stack of 4 (48 by 48) frames
        input_0 = layers.Input(shape=(48, 48, 4))

        # Main network backbone. This is shared by the actor and critic.
        conv_0 = create_conv_layer(8, 4, 2)(input_0)
        conv_1 = create_conv_layer(16, 3, 2)(conv_0)
        conv_2 = create_conv_layer(32, 3, 2)(conv_1)
        conv_3 = create_conv_layer(64, 3, 1)(conv_2)
        conv_4 = create_conv_layer(128, 3, 1)(conv_3)
        flat_0 = layers.Flatten()(conv_4)

        # Actor output
        dense_0 = layers.Dense(128, activation='relu')(flat_0)
        dense_1 = layers.Dense(6, activation='softplus')(dense_0)
        reshape_0 = layers.Reshape((3, 2))(dense_1)
        lamb_0 = layers.Lambda(lambda x: x + 1)(reshape_0)  # Ensure alpha and beta are > 1

        # Critic output
        dense_2 = layers.Dense(128, activation='relu')(flat_0)
        dense_3 = layers.Dense(1)(dense_2)

        # Compile model
        model = tf.keras.Model(inputs=[input_0], outputs=[lamb_0, dense_3])
        model.compile(optimizer=tf.optimizers.Adam(1e-3))

        return model


class CustomCarRacing(CarRacing):
    """
    This is a wrapper around the standard 'CarRacing' environment. It modifies the behavior of the step(), including changes to the observation space and changes to the reward
    amount. This environment wrapper is intended to be used with the PPO agent.
    """

    def __init__(self, image_stack_size=4, step_repeat=8):
        """
        Create a new environment.
        :param image_stack_size:
        :param step_repeat: The number of times to repeat internal calls to step() for every external call. This greatly speeds up training since each frame is very similar. From
        the outside it will appear as if the environment is running at FPS / STEP_REPEAT frames per second, where FPS is the original FPS o
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

        image_stack_array = np.empty((48, 48, 4))
        for i in range(self._image_stack_size):
            image_stack_array[..., i] = self._image_stack[i]

        return image_stack_array, total_reward, done, info

    @staticmethod
    def _preprocess_observation(observation):

        # Keep only the green channel of the RGB image and reduce the resolution
        observation = observation[::2, ::2, 1]

        # Convert to float
        observation = observation.astype(np.float32)

        # Normalize values between -1 and 1
        observation = (observation / 128) - 1

        return observation
