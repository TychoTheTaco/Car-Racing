import tensorflow as tf
import tensorflow_probability
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow_probability.python.layers.distribution_layer import DistributionLambda

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

a = np.empty((5, 1, 1, 1))
print(a[0])
print(a.ravel())


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
        observation = observation.astype(np.float32)
        observation = observation / 128 - 1
        return observation


class ProximalPolicyOptimizationAgent(Agent):

    def __init__(self, env: gym.Env):

        self._env = env

        def create_conv_layer(filters, kernel_size, strides):
            return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', kernel_initializer=tf.initializers.glorot_normal(), bias_initializer=tf.initializers.constant(0.1))

        i0 = layers.Input(shape=(96, 96, 4))
        c0 = create_conv_layer(8, 4, 2)(i0)
        c1 = create_conv_layer(16, 3, 2)(c0)
        c2 = create_conv_layer(32, 3, 2)(c1)
        c3 = create_conv_layer(64, 3, 2)(c2)
        c4 = create_conv_layer(128, 3, 1)(c3)
        c5 = create_conv_layer(256, 3, 1)(c4)
        #f0 = layers.Flatten()(c5)

        d0 = layers.Dense(100, activation='relu')(c5)
        a0 = layers.Dense(3, activation='softplus')(d0)
        b0 = layers.Dense(3, activation='softplus')(d0)
        print(a0)
        print(b0)
        cc = layers.Concatenate()([a0, b0])
        print(cc)
        r0 = layers.Reshape((2, 3))(cc)
        t0 = layers.Permute((2, 1))(r0)
        print(t0)
        #def mdf(inputs):
        #    print(inputs)
        #    return Beta(inputs[0], inputs[1])
        #o0 = DistributionLambda(make_distribution_fn=mdf)([a0, b0])

        d1 = layers.Dense(100, activation='relu')(c5)
        d2 = layers.Dense(1, activation='relu')(d1)
        o1 = layers.Flatten()(d2)

        self._model = tf.keras.Model(inputs=[i0], outputs=[t0, o1])
        self._model.compile(optimizer=tf.optimizers.Adam(0.001))
        self._model.summary()


    def get_action(self, observation, action_space: gym.Space):
        pass

    def train(self, episodes: int = 1000):

        # Keep track of some stats
        episode_rewards = []

        BUFFER_SIZE = 2000  # TODO: 2000
        BATCH_SIZE = 128
        GAMMA = 0.99

        PPO_EPOCH = 10
        CLIP_PARAM = 0.1

        max_a_log_p = 0
        max_alpha, max_beta = np.zeros((3,)), np.zeros((3,))
        min_action, max_action = np.ones((3,)), np.zeros((3,))

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

                if lp is np.inf:
                    print('INFINITY:', alpha, beta, action)
                    exit(1)

                min_action = np.minimum(min_action, action)
                max_action = np.maximum(max_action, action)
                #print(min_action, max_action)

                max_a_log_p = max(max_a_log_p, lp)
                #print(max_a_log_p)
                max_alpha = np.maximum(max_alpha, alpha)
                max_beta = np.maximum(max_beta, beta)
                #print(max_alpha, max_beta)

                A = action.numpy()
                A[0] = np.interp(A[0], [0, 1], [-1, 1])

                # Perform action
                new_observation, reward, done, _ = self._env.step(A)
                episode_reward += reward
                reward_history.append(reward)

                if reward > 100:
                    print('REWARD TOO HIGH:', reward)
                    print('E:', episode)
                    print('A:', action)
                    print('AB:', alpha, beta)
                    print('LP:', lp)
                    print(min_action, max_action)
                    print(max_alpha, max_beta)
                    print(max_a_log_p)
                    exit(1)

                # End early if we haven't earned any reward for a while
                if any([x > 0 for x in reward_history[:-100]]):
                    done = True
                    print('DONE EARLY!')

                transitions.append((observation, action, lp, reward, new_observation))
                if len(transitions) == BUFFER_SIZE:

                    s = tf.convert_to_tensor(([x[0] for x in transitions]))
                    a = tf.convert_to_tensor([x[1] for x in transitions])
                    old_a_logp = tf.convert_to_tensor([x[2] for x in transitions])
                    r = tf.convert_to_tensor([x[3] for x in transitions])
                    s_ = tf.convert_to_tensor([x[4] for x in transitions])

                    print('[S] MIN:', tf.reduce_min(s).numpy(), 'MAX:', tf.reduce_max(s).numpy())
                    print('[A] MIN:', tf.reduce_min(a).numpy(), 'MAX:', tf.reduce_max(a).numpy())
                    print('[L] MIN:', tf.reduce_min(old_a_logp).numpy(), 'MAX:', tf.reduce_max(old_a_logp).numpy())
                    print('[R] MIN:', tf.reduce_min(r).numpy(), 'MAX:', tf.reduce_max(r).numpy())
                    print('[N] MIN:', tf.reduce_min(s_).numpy(), 'MAX:', tf.reduce_max(s_).numpy())

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


if __name__ == '__main__':
    env = gym.wrappers.TimeLimit(CustomCarRacing(), 1000)

    ppo = ProximalPolicyOptimizationAgent(env)
    ppo.train()
