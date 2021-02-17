import tensorflow as tf
import tensorflow_probability
from tensorflow_probability.python.distributions.beta import Beta

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from tensorflow.keras import layers
import gym
from agents import Agent
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from car_racing_environment import CarRacing


class CustomCarRacing(CarRacing):

    def __init__(self):
        super().__init__(verbose=0)
        self._image_stack = []

    def reset(self):
        self._image_stack.clear()
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)

        observation = self._transform_observation(observation)

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

    def _transform_observation(self, observation):
        observation = observation[:, :, 1]
        return observation


class ProximalPolicyOptimizationAgent(Agent):

    def __init__(self, env: gym.Env):

        self._env = env

        def create_conv_layer(filters, kernel_size, strides):
            return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', weights=tf.initializers.glorot_normal(), bias_initializer=tf.initializers.constant(0.1))

        # Create Actor network
        self._actor = tf.keras.Sequential([
            layers.InputLayer(input_shape=(96, 96, 4)),
            create_conv_layer(8, 4, 2),
            create_conv_layer(16, 3, 2),
            create_conv_layer(32, 3, 2),
            create_conv_layer(64, 3, 1),
            create_conv_layer(128, 3, 1),
            #layers.Conv2D(8, kernel_size=4, strides=2, activation='relu', weights=tf.initializers.glorot_normal(), bias_initializer=tf.initializers.constant(0.1)),
            #layers.Conv2D(16, kernel_size=3, strides=2, activation='relu'),
            #layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
            #layers.Conv2D(64, kernel_size=3, activation='relu'),
            #layers.Conv2D(128, kernel_size=3, activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(6, activation='softplus'),
            layers.Reshape((3, 2))
        ])
        self._actor.compile(optimizer=tf.optimizers.Adam(0.001))

        # Create Critic network
        i0 = layers.Input(shape=(96, 96, 4))
        f0 = layers.Flatten()(i0)
        i1 = layers.Input(shape=(3,))
        f1 = layers.Flatten()(i1)
        c0 = tf.keras.layers.Concatenate()([f0, f1])
        d0 = tf.keras.layers.Dense(256, activation='relu')(c0)
        d1 = tf.keras.layers.Dense(1, activation='linear')(d0)
        self._critic = tf.keras.Model(inputs=[i0, i1], outputs=d1)

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
        MAX_GRAD_NORM = 0.5

        transitions = []

        for episode in range(1, episodes + 1):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0

            done = False
            while not done:

                self._env.render()

                # Choose action
                p = self._actor.predict(np.expand_dims(observation, axis=0))[0]
                beta_distribution = Beta(p[:, 0], p[:, 1])
                action = beta_distribution.sample().numpy()
                lp = beta_distribution.log_prob(action)  # action log probability
                lp = tf.reduce_sum(lp).numpy()
                action[0] = np.interp(action[0], [0, 1], [-1, 1])

                # Perform action
                new_observation, reward, done, _ = self._env.step(action)
                episode_reward += reward

                transitions.append((observation, action, lp, reward, new_observation))
                if len(transitions) == BUFFER_SIZE:

                    s = np.array([x[0] for x in transitions])
                    a = np.array([x[1] for x in transitions])
                    old_a_logp = np.array([x[2] for x in transitions])
                    r = np.array([x[3] for x in transitions])
                    s_ = np.array([x[4] for x in transitions])

                    target_v = r + GAMMA * self._critic.predict(s_)
                    adv = target_v - self._critic.predict(s)

                    for _ in range(PPO_EPOCH):
                        indexes = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)

                        p = self._actor.predict(np.expand_dims(s[indexes], axis=0))[0]
                        alphas, betas = p[:, 0], p[:, 1]
                        dist = Beta(alphas, betas)
                        a_log_p = dist.log_prob(a[indexes])
                        a_log_p = tf.reduce_sum(a_log_p, axis=1, keepdims=True)
                        ratio = np.exp(a_log_p - old_a_logp[indexes])

                        surr1 = ratio * adv[indexes]
                        surr2 = tf.clip_by_value(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM) * adv[indexes]
                        action_loss = -tf.minimum(surr1, surr2)
                        action_loss = tf.reduce_mean(action_loss)
                        value_loss = tf.losses.huber(self._critic.predict(s[indexes]), target_v[indexes])
                        loss = action_loss + 2 * value_loss

                    transitions.clear()

                observation = new_observation

            episode_rewards.append(episode_reward)
            print(f'Episode {episode} | Reward: {int(episode_reward)} | Average: {np.average(episode_rewards[-20:]):.02f}')


if __name__ == '__main__':
    env = gym.wrappers.TimeLimit(CustomCarRacing(), 1000)

    ppo = ProximalPolicyOptimizationAgent(env)
    ppo.train()
