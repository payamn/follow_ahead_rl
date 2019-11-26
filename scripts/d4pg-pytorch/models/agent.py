import shutil
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import torch

from utils.utils import OUNoise, make_gif
from utils.logger import Logger
from env.utils import create_env_wrapper


class Agent(object):

    def __init__(self, config, policy, global_episode, n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.env_wrapper.env.set_agent(self.n_agent)
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])
        self.ou_noise.reset()

        self.actor = policy

        # Logger
        log_path = f"{log_dir}/agent-{n_agent}"
        self.logger = Logger(log_path)

    def update_actor_learner(self, learner_w_queue):
        """Update local actor to the actor from learner. """
        if learner_w_queue.empty():
            return
        source = learner_w_queue.get()
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()
            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            print("call reset on agent {}".format(self.n_agent))
            state = self.env_wrapper.reset()
            print("called reset on agent {}".format(self.n_agent))
            self.ou_noise.reset()
            self.env_wrapper.env.resume_simulator()
            done = False
            while not done:
                print ("state is {} agent {}".format(state, self.n_agent))
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action, num_steps)
                    action = action.squeeze(0)
                else:
                    action = action.detach().cpu().numpy().flatten()
                next_state, reward, done = self.env_wrapper.step(action)

                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']
                    if not replay_queue.full():
                        replay_queue.put([state_0, action_0, discounted_reward, next_state, done, gamma])

                state = next_state

                if done or num_steps == self.max_steps:
                    print ("agent {} done".format(self.n_agent))
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        print("agent {} exp_buffer_len {}".format(self.n_agent, len(self.exp_buffer)))
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            print("agent {} exp_buffer_len {}".format(self.n_agent, len(self.exp_buffer)))
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        replay_queue.put([state_0, action_0, discounted_reward, next_state, done, gamma])
                    break

                num_steps += 1

            print("agent {} finished if".format(self.n_agent))
            # Log metrics
            step = update_step.value
            self.logger.scalar_summary("agent/reward", episode_reward, step)
            self.logger.scalar_summary("agent/episode_timing", time.time() - ep_start_time, step)

            # Saving agent
            if self.local_episode % self.num_episode_save == 0 or episode_reward > best_reward:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")
                print("reward is: {} step: {} ".format(episode_reward, step))

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_actor_learner(learner_w_queue)

            print ("done statring next episode agent: {}".format(self.n_agent))
        while not replay_queue.empty():
            replay_queue.get()

        # Save replay from the first agent only
        # if self.n_agent == 0:
        #    self.save_replay_gif()

        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        last_path = f"{self.log_dir}"
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        if not os.path.exists(last_path):
            os.makedirs(last_path)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)
        model_fn = f"{last_path}/best.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self):
        dir_name = "replay_render"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        self.env_wrapper.env.resume_simulator()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")