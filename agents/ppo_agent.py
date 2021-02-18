import tensorflow as tf
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from tensorflow_probability.python.distributions.beta import Beta
from tensorflow.keras import layers
import gym
from agents import Agent
import numpy as np
from pathlib import Path
from typing import Union, Optional
from car_racing_environment import CarRacing


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
        pass

    def train(self, episodes: int = 1000):

        # Keep track of some stats
        episode_rewards = []

        BUFFER_SIZE = 2000
        BATCH_SIZE = 128
        GAMMA = 0.99

        PPO_EPOCH = 10
        CLIP_PARAM = 0.1

        transitions = []

        for episode in range(1, episodes + 1):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0
            reward_history = []

            done = False
            while not done:

                #self._env.render()

                # Choose action
                p = self._model.predict(np.expand_dims(observation, axis=0))[0][0]
                alpha, beta = p[:, 0] + 1, p[:, 1] + 1  # TODO : alpha, beta sometimes too large? maybe
                beta_distribution = Beta(alpha, beta)
                action = beta_distribution.sample()
                lp = beta_distribution.log_prob(action)  # action log probability
                lp = tf.reduce_sum(lp).numpy()

                A = action.numpy()
                A[0] = np.interp(A[0], [0, 1], [-1, 1])

                # Perform action
                new_observation, reward, done, _ = self._env.step(A)
                episode_reward += reward
                reward_history.append(reward)

                # End early if we haven't earned any reward for a while
                if any([x > 0 for x in reward_history[:-100]]):
                    done = True
                    print('DONE EARLY!')

                transitions.append((observation, action, lp, reward, new_observation))
                if len(transitions) == BUFFER_SIZE:

                    s = tf.convert_to_tensor([x[0] for x in transitions])
                    a = tf.convert_to_tensor([x[1] for x in transitions])
                    old_a_logp = tf.convert_to_tensor([x[2] for x in transitions])
                    r = tf.convert_to_tensor([x[3] for x in transitions])
                    s_ = tf.convert_to_tensor([x[4] for x in transitions])

                    target_v = r + GAMMA * self._model.predict(s_)[1].ravel()
                    adv = target_v - self._model.predict(s)[1].ravel()

                    for _ in range(PPO_EPOCH):
                        indexes = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)

                        with tf.GradientTape() as tape:

                            p = self._model(tf.gather(s, indexes))[0]
                            alphas, betas = p[:, :, 0], p[:, :, 1]
                            dist = Beta(alphas, betas)
                            a_log_p = dist.log_prob(tf.gather(a, indexes))
                            a_log_p = tf.reduce_sum(a_log_p, axis=1, keepdims=True)
                            ratio = tf.exp(a_log_p - tf.expand_dims(tf.gather(old_a_logp, indexes), axis=1))

                            surr1 = ratio * tf.expand_dims(tf.gather(adv, indexes), axis=1)
                            surr2 = tf.clip_by_value(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * tf.expand_dims(tf.gather(adv, indexes), axis=1)
                            action_loss = tf.reduce_mean(-tf.minimum(surr1, surr2))
                            value_loss = tf.losses.huber(tf.gather(target_v, indexes), tf.reshape(self._model(tf.gather(s, indexes))[1], (BATCH_SIZE,)))  # True vs Predicted
                            loss = action_loss + 2 * value_loss

                        g = tape.gradient(loss, self._model.trainable_variables)
                        # for x in g:
                        #     if x is None:
                        #         continue
                        #     m0 = tf.reduce_max(x)
                        #     print('GMAX:', m0)
                        self._model.optimizer.apply_gradients(zip(g, self._model.trainable_variables))

                    transitions.clear()

                observation = new_observation

            episode_rewards.append(episode_reward)
            print(f'Episode {episode} | Reward: {int(episode_reward)} | Average: {np.average(episode_rewards[-20:]):.02f}')

    def _create_model(self):

        def create_conv_layer(filters, kernel_size, strides):
            return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', kernel_initializer=tf.initializers.glorot_normal(), bias_initializer=tf.initializers.constant(0.1))

        # Main backbone
        input_0 = layers.Input(shape=(96, 96, 4))
        conv_0 = create_conv_layer(8, 4, 2)(input_0)
        conv_1 = create_conv_layer(16, 3, 2)(conv_0)
        conv_2 = create_conv_layer(32, 3, 2)(conv_1)
        conv_3 = create_conv_layer(64, 3, 2)(conv_2)
        conv_4 = create_conv_layer(128, 3, 1)(conv_3)
        conv_5 = create_conv_layer(256, 3, 1)(conv_4)
        flatten_0 = layers.Flatten()(conv_5)

        # Actor output
        dense_0 = layers.Dense(100, activation='relu')(flatten_0)
        alpha_output = layers.Dense(3, activation='softplus')(dense_0)
        beta_output = layers.Dense(3, activation='softplus')(dense_0)
        concat_0 = layers.Concatenate()([alpha_output, beta_output])
        reshape_0 = layers.Reshape((2, 3))(concat_0)
        output_0 = layers.Permute((2, 1))(reshape_0)

        # Critic output
        dense_1 = layers.Dense(100, activation='relu')(flatten_0)
        dense_2 = layers.Dense(1, activation='relu')(dense_1)
        output_1 = layers.Flatten()(dense_2)

        model = tf.keras.Model(inputs=[input_0], outputs=[output_0, output_1])
        model.compile(optimizer=tf.optimizers.Adam(1e-3))

        return model


class CustomCarRacing(CarRacing):

    def __init__(self):
        super().__init__(verbose=0)
        self._image_stack = []

    def reset(self):
        self._image_stack.clear()
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)

        observation = self._preprocess_observation(observation)

        self._image_stack.append(observation)

        if len(self._image_stack) < 4:
            observation = np.empty((96, 96, 4))
            observation[:, :, 0] = self._image_stack[-1]
            observation[:, :, 1] = self._image_stack[-1]
            observation[:, :, 2] = self._image_stack[-1]
            observation[:, :, 3] = self._image_stack[-1]
        else:
            observation = np.empty((96, 96, 4))
            observation[:, :, 0] = self._image_stack[0]
            observation[:, :, 1] = self._image_stack[1]
            observation[:, :, 2] = self._image_stack[2]
            observation[:, :, 3] = self._image_stack[3]
            self._image_stack.pop(0)

        return observation, reward, done, info

    def _preprocess_observation(self, observation):

        # Keep only the green channel of the RGB image.
        observation = observation[:, :, 1]

        # Convert to float
        observation = observation.astype(np.float32)

        # Normalize values between -1 and 1
        observation = observation / 128 - 1

        return observation