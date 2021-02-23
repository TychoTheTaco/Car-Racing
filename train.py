import datetime
from collections import deque
import tensorflow as tf

tf.random.set_seed(0)
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

        # A deque is used to limit the size of the image stack. It is converted to a numpy array before being returned from step(). This array is reused for every call.
        self._image_stack = deque(maxlen=self._image_stack_size)
        self._image_stack_array = np.empty((48, 48, self._image_stack_size))

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
        for i in range(self._image_stack_size):
            self._image_stack_array[:, :, i] = self._image_stack[-1 if len(self._image_stack) < self._image_stack_size else i]

        return self._image_stack_array, total_reward, done, info

    def _preprocess_observation(self, observation):

        # Keep only the green channel of the RGB image and reduce the resolution
        observation = observation[::2, ::2, 1]

        # Convert to float
        observation = observation.astype(np.float32)

        # Normalize values between -1 and 1
        observation = (observation / 128) - 1

        return observation




def create_model():
    
    def create_conv_layer(filters, kernel_size, strides):
        return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu', kernel_initializer=tf.initializers.glorot_normal(),
                             bias_initializer=tf.initializers.constant(0.1))

    # Input is a stack of 4 (48 by 48) frames
    input_0 = layers.Input(shape=(48, 48, 4))

    # Main network backbone. This is shared by the actor and critic.
    conv_0 = create_conv_layer(8, 3, 2)(input_0)
    conv_1 = create_conv_layer(16, 3, 2)(conv_0)
    conv_2 = create_conv_layer(32, 3, 2)(conv_1)
    conv_3 = create_conv_layer(64, 3, 1)(conv_2)
    flat_0 = layers.Flatten()(conv_3)

    # Actor output
    dense_0 = layers.Dense(128, activation='relu')(flat_0)
    dense_1 = layers.Dense(6, activation='softplus')(dense_0)
    reshape_0 = layers.Reshape((3, 2))(dense_1)
    lamb_0 = layers.Lambda(lambda x: x + 1)(reshape_0)  # Ensure alpha and beta are > 1

    # Critic output
    dense_2 = layers.Dense(128, activation='relu')(flat_0)
    dense_3 = layers.Dense(1, activation='relu')(dense_2)

    # Compile model
    model = tf.keras.Model(inputs=[input_0], outputs=[lamb_0, dense_3])
    model.compile(optimizer=tf.optimizers.Adam(1e-3))

    return model


def gather_experience(buffer_size):
    
    # Create environment
    env = CustomCarRacing()
    
    transitions = []
    
    while True:

        # Reset environment
        observation = env.reset()
        episode_reward = 0
        reward_history = []
    
        done = False
        while not done:
    
            # Choose action
            p = model.predict(np.expand_dims(observation, axis=0))[0][0]
            alpha, beta = p[:, 0], p[:, 1]
            beta_distribution = Beta(alpha, beta)
            action = beta_distribution.sample()
            lp = beta_distribution.log_prob(action)
            lp = tf.reduce_sum(lp)
    
            A = action.numpy()
            A[0] = np.interp(A[0], [0, 1], [-1, 1])
    
            # Perform action
            new_observation, reward, done, _ = env.step(A)
            episode_reward += reward
            reward_history.append(reward)
    
            # End early if we haven't earned any reward for a while
            if np.mean(reward_history[-100:]) < -0.2:
                done = True
    
            transitions.append((observation, action, lp, reward, new_observation))
            if len(transitions) >= buffer_size:
                # give transitions to main thread
                # wait for new model (maybe this can just be global)
                env.close()
                return transitions
    
            observation = new_observation

        # Print some statistics
        print(f'Episode {episode} | Reward: {episode_reward:.02f} | Average: {np.average(episode_rewards[-50:]):.02f}')


if __name__ == '__main__':
    
    model_checkpoint_dir = 'models'
    episodes = 1000
    log_interval = 10
    save_interval = 50
    
    model_dir = Path(model_checkpoint_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    # Keep track of some stats
    episode_rewards = []

    WORKERS = 4
    BUFFER_SIZE = 2000 // WORKERS
    BATCH_SIZE = 128
    GAMMA = 0.99

    PPO_EPOCH = 10
    CLIP_PARAM = 0.1
    
    model = create_model()

    for episode in range(episodes):

        transitions = []
        for actor in range(WORKERS):
            ts = gather_experience(BUFFER_SIZE)
            transitions.extend(ts)
        print('what the ')

        # transitions = []
        #
        # from concurrent import futures
        # with futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(gather_experience, BUFFER_SIZE) for _ in range(WORKERS)]
        #     results = [future.result() for future in futures]
        #
        #     for r in results:
        #         transitions.extend(r)
        #
        print('learning!')

        states = tf.convert_to_tensor([x[0] for x in transitions])
        actions = tf.convert_to_tensor([x[1] for x in transitions])
        old_a_logp = tf.convert_to_tensor([x[2] for x in transitions])
        rewards = tf.convert_to_tensor([x[3] for x in transitions])
        new_states = tf.convert_to_tensor([x[4] for x in transitions])

        old_a_logp = tf.expand_dims(old_a_logp, axis=1)
        r = tf.expand_dims(rewards, axis=1)

        target_v = r + GAMMA * model(new_states)[1]
        adv = target_v - model(states)[1]

        for _ in range(PPO_EPOCH):
            indexes = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)

            with tf.GradientTape() as tape:
                ab = model(tf.gather(states, indexes))[0]
                alpha, beta = ab[:, :, 0], ab[:, :, 1]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(tf.gather(actions, indexes))
                a_logp = tf.reduce_sum(a_logp, axis=1, keepdims=True)
                ratio = tf.exp(a_logp - tf.gather(old_a_logp, indexes))
                surr1 = ratio * tf.gather(adv, indexes)
                surr2 = tf.clip_by_value(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * tf.gather(adv, indexes)
                action_loss = -tf.minimum(surr1, surr2)
                action_loss = tf.reduce_mean(action_loss)

                one = model(tf.gather(states, indexes))[1]
                two = tf.gather(target_v, indexes)
                value_loss = tf.losses.mse(two, one)
                value_loss = tf.reduce_mean(value_loss)

                loss = action_loss + value_loss

            g = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(g, model.trainable_variables))
