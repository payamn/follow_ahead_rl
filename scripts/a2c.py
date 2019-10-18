import time
import gym
import os
import gym_gazebo
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import argparse


class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.conv_1 = kl.Conv2D(32, (15, 15), activation='relu', input_shape=(50, 100, 5))
        self.batch_norm1 = kl.BatchNormalization()
        self.conv_2 = kl.Conv2D(32, (12, 12), activation='relu')
        self.batch_norm2 = kl.BatchNormalization()
        self.conv_3 = kl.Conv2D(32, (10, 10), activation='relu')
        self.batch_norm3 = kl.BatchNormalization()
        self.conv_4 = kl.Conv2D(32, (8, 8), activation='relu')
        self.batch_norm4 = kl.BatchNormalization()
        self.fc_image = kl.Dense(256, activation='relu')

        self.fc1 = kl.Dense(32, activation='relu')
        self.fc2 = kl.Dense(64, activation='relu')
        self.fc3 = kl.Dense(128, activation='relu')
        self.fc4 = kl.Dense(256, activation='relu')

        self.concat = kl.Concatenate(axis=-1)
        self.hidden1 = kl.Dense(128, activation='relu' )
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        self.flatten = kl.Flatten()
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x1 = tf.convert_to_tensor(inputs[0])
        x1 = tf.dtypes.cast(x1, tf.float32)
        x2 = tf.convert_to_tensor(inputs[1])

        # separate hidden layers from the same input tensor
        out_image =  self.conv_1(x1)
        out_image = self.batch_norm1(out_image)
        out_image = self.conv_2(out_image)
        out_image = self.batch_norm2(out_image)
        out_image = self.conv_3(out_image)
        out_image = self.batch_norm3(out_image)
        out_image = self.conv_4(out_image)
        out_image = self.batch_norm4(out_image)
        flatt = self.flatten(out_image)
        flatt = self.fc_image(flatt)

        # tf.print(tf.shape(flatt))
        out_person = self.fc1(x2)
        out_person = self.fc2(out_person)
        out_person = self.fc3(out_person)
        out_person = self.fc4(out_person)

        flatt = self.concat([out_person, flatt])
        hidden_logs = self.hidden1(flatt)
        hidden_vals = self.hidden2(flatt)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, scan_img_obs, pos_vel_obs):
        # executes call() under the hood
        logits, value = self.predict((scan_img_obs, pos_vel_obs))
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.25,
            'entropy': 0.01
        }
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0004),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )


    def train(self, env, batch_sz=200, updates=1000000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = [np.empty((batch_sz,) + env.observation_space[0].shape ),
                        np.empty((batch_sz,) + env.observation_space[1].shape)]
        ep_rews = [0.0]
        env.resume_simulator()
        next_obs = env.get_observation()
        best_losses = float('inf')
        first = True
        for update in range(updates):
            # next_obs, rewards[0], dones[0], _ = env.step(6)
            # continue
            env.resume_simulator()
            for step in range(batch_sz):
                observations[0][step] = next_obs[0].copy()
                observations[1][step] = next_obs[1].copy()
                actions[step], values[step] = self.model.action_value(next_obs[0][None, :], next_obs[1][None, :])
                # time.sleep(0.1)
                if first:
                    print(self.model.summary()) # training loop: collect samples, send to optimizer, repeat updates times
                    first = False
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                print(actions[step])
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    print("before reset")
                    next_obs = env.reset()
                    print("reset done")
                    env.resume_simulator()
                    logging.info("Episode: %03d, Reward: %03d " % (len(ep_rews) - 1, ep_rews[-2]))
                    print("Episode: %d, Reward: %f" % (len(ep_rews) - 1, ep_rews[-2]))
                # print (step, actions[step])

            env.pause()
            _, next_value = self.model.action_value(next_obs[0][None, :], next_obs[1][None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it

            losses = self.model.train_on_batch((observations[0], observations[1]), [acts_and_advs, returns])
            if losses[0] < best_losses:
                model.save_weights("weights/bestloses")
                best_losses = losses[0]
            elif update % 100 == 0:
                model.save_weights("weights/"+str(update))
            print (rewards, dones, losses)
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))
        return ep_rews

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy'] * entropy_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input weight file of the network')
    parser.add_argument('--weight', default="weights/bestloses", type=str, help='weight file')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print (gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except Exception as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    print ("before env")
    env = gym.make('gazebo-v0')
    print ("after env")
    model = Model(num_actions=env.action_space.n)

    if args.weight is not None and os.path.exists(args.weight+".index"):
        model.load_weights(args.weight)
        print("weight loaded:", args.weight)
    else:
        print("weight not loaded:", args.weight)

    agent = A2CAgent(model)

    rewards_history = agent.train(env)
    print("Finished training.")
    print("Total Episode Reward: %d out of 200" % agent.test(env, True))

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
