import datetime
import tensorflow as tf
from tensorflow_probability.python.distributions.beta import Beta
from tensorflow.keras import layers
import gym
from agents import Agent
import numpy as np
from pathlib import Path
from typing import Union, Optional
import matplotlib.pyplot as plt


# Disable all GPUs
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

tf.random.set_seed(0)
np.random.seed(0)


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
        p = self._model.predict(np.expand_dims(observation, axis=0))[0][0]
        alpha, beta = p[:, 0], p[:, 1]
        distribution = Beta(alpha, beta)
        action = distribution.sample().numpy()
        action[0] = np.interp(action[0], [0, 1], [-1, 1])
        return action

    def train(self,
              episodes: int = 1000,
              log_interval: int = 10,
              model_dir: str = 'models',
              save_interval: int = 100,
              buffer_size: int = 2000,
              batch_size: int = 128,
              gamma: float = 0.99,
              ppo_epochs: int = 10,
              clip_epsilon: float = 0.1):
        model_dir = Path(model_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        model_dir.mkdir(parents=True, exist_ok=True)

        training_start_time = datetime.datetime.now()
        print('Started training at', training_start_time.strftime('%d-%m-%Y %H:%M:%S'))

        # Keep track of some stats
        episode_rewards = []
        moving_average_range = 50

        transitions = []

        for episode in range(episodes):

            # Reset environment
            observation = self._env.reset()
            episode_reward = 0

            done = False
            while not done:

                # Choose action
                p = self._model(np.expand_dims(observation, axis=0))[0][0]
                alpha, beta = p[:, 0], p[:, 1]
                beta_distribution = Beta(alpha, beta)
                action = beta_distribution.sample()
                log_prob = tf.reduce_sum(beta_distribution.log_prob(action))

                A = action.numpy()
                A[0] = np.interp(A[0], [0., 1.], [-1., 1.])

                # Perform action
                new_observation, reward, done, _ = self._env.step(A)
                episode_reward += reward

                transitions.append((observation, action, log_prob, reward, new_observation))
                if len(transitions) >= buffer_size:
                    print('learning!')

                    states = tf.convert_to_tensor([x[0] for x in transitions])
                    actions = tf.convert_to_tensor([x[1] for x in transitions])
                    old_a_logp = tf.expand_dims(tf.convert_to_tensor([x[2] for x in transitions]), axis=1)
                    rewards = tf.expand_dims(tf.convert_to_tensor([x[3] for x in transitions]), axis=1)
                    new_states = tf.convert_to_tensor([x[4] for x in transitions])

                    target_v = rewards + gamma * self._model(new_states)[1]
                    adv = target_v - self._model(states)[1]

                    def gen_batches(indices, batch_size):
                        for i in range(0, len(indices), batch_size):
                            yield indices[i:i + batch_size]

                    for _ in range(ppo_epochs):
                        indices = np.arange(buffer_size)
                        np.random.shuffle(indices)

                        for batch in gen_batches(indices, batch_size):

                            with tf.GradientTape() as tape:

                                # Calculate action loss
                                ab = self._model(tf.gather(states, batch))[0]
                                alpha, beta = ab[:, :, 0], ab[:, :, 1]
                                dist = Beta(alpha, beta)
                                a_logp = tf.reduce_sum(dist.log_prob(tf.gather(actions, batch)), axis=1, keepdims=True)
                                ratio = tf.exp(a_logp - tf.gather(old_a_logp, batch))
                                surr1 = ratio * tf.gather(adv, batch)
                                surr2 = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * tf.gather(adv, batch)
                                action_loss = tf.reduce_mean(-tf.minimum(surr1, surr2))

                                # Calculate value loss
                                one = self._model(tf.gather(states, batch))[1]
                                two = tf.gather(target_v, batch)
                                value_loss = tf.reduce_mean(tf.losses.mse(two, one))

                                # Calculate combined loss
                                loss = action_loss + 2 * value_loss

                            g = tape.gradient(loss, self._model.trainable_variables)
                            self._model.optimizer.apply_gradients(zip(g, self._model.trainable_variables))

                    transitions.clear()

                observation = new_observation

            episode_rewards.append(episode_reward)

            # Print some statistics
            if not episode % log_interval:
                print(f'Episode {episode} | Reward: {episode_reward:.02f} | Moving Average: {np.average(episode_rewards[-50:]):.02f}')

            # Save model
            if not episode % save_interval:
                self._model.save(model_dir / f'episode-{episode}.h5')

        # Save final model
        self._model.save(model_dir / 'model.h5')

        training_end_time = datetime.datetime.now()
        print('Finished training at', training_end_time.strftime('%d-%m-%Y %H:%M:%S'))
        print('Total training time:', training_end_time - training_start_time)
        np.savetxt(model_dir / 'rewards.txt', episode_rewards)

        # Plot statistics
        x_axis = np.arange(len(episode_rewards))
        plt.figure(1, figsize=(16, 9))
        plt.plot(x_axis, episode_rewards, label='Episode reward')
        moving_averages = [np.mean(episode_rewards[i - (moving_average_range - 1):i + 1]) if i >= (moving_average_range - 1) else np.mean(episode_rewards[:i + 1]) for i in range(len(episode_rewards))]
        plt.plot(x_axis, moving_averages, color='red', label=f'{moving_average_range}-episode moving average')
        plt.title('Training Performance')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend(loc='upper left')
        plt.savefig(model_dir / 'rewards.jpg')
        plt.show()

    def _create_model(self):

        def create_conv_layer(filters, kernel_size, strides):
            return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                                 bias_initializer=tf.initializers.constant(0.1))

        # Input is a stack of frames
        input_0 = layers.Input(shape=(32, 32, 4))

        # Main network backbone. This is shared by the actor and critic.
        conv_0 = create_conv_layer(8, 4, 2)(input_0)
        conv_1 = create_conv_layer(16, 3, 2)(conv_0)
        conv_2 = create_conv_layer(32, 3, 2)(conv_1)
        conv_3 = create_conv_layer(64, 3, 1)(conv_2)
        flat_0 = layers.Flatten()(conv_3)

        # Actor output
        dense_0 = layers.Dense(64, activation='relu')(flat_0)
        dense_1 = layers.Dense(6, activation='softplus')(dense_0)
        reshape_0 = layers.Reshape((3, 2))(dense_1)
        lamb_0 = layers.Lambda(lambda x: x + 1)(reshape_0)  # Ensure alpha and beta are > 1

        # Critic output
        dense_2 = layers.Dense(64, activation='relu')(flat_0)
        dense_3 = layers.Dense(1)(dense_2)

        # Compile model
        model = tf.keras.Model(inputs=[input_0], outputs=[lamb_0, dense_3])
        model.compile(optimizer=tf.optimizers.Adam(0.001))

        return model
