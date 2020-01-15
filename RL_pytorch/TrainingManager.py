#!/usr/bin/env python
""" 
    Training Manager. It takes an agent, environement and run the scenario for a set of episodes.
    It show online learning  curve
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

from AbstractAgent import AbstractAgent
from AbstractEnvironment import AbstractEnvironement

from collections import deque # to mantian the last k rewards

class TrainingManager:
    def __init__(self,
                 num_episodes,
                 episode_length,
                 agent:AbstractAgent,
                 env:AbstractEnvironement,
                 average_reward_steps=5,
                 device='cpu',
                 logfile='training_log.txt'):

        """TrainingManager initializer.

        Keyword arguments:
        num_episodes -- number of episodes
        episode_length -- length of an episode
        agent -- a RL agent, see AbstractAgent
        env  -- an environement object, see AbstractEnvironment 
        average_reward_steps -- number of last episodes where the average reward is calculated from
        device -- where will the neural networks be trained ("cpu", "gpu"). If multiple gpus exist the manager will choose the first avialable.
        """

        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.agent = agent
        self.env = env
        self.average_reward_steps = average_reward_steps
        self.device = device

    def run(self, verbose=False, plot=False, save_to_file=True, parallel=False):
        """ Run the RL scenario using the settings of the TrainingManager
        
        
        Keyword arguments:
        verbose -- if True, the manager will print the total reward for each epsiode and some other statics
        plot -- if True, the manager will plot online learning curve
        parallel -- if True the manager will parallel - NOT SUPPORTED YET.
        """
        
        # do some assertions first
        assert self.agent is not None, "Agent can not be None"
        assert self.env is not None, "Environment object can not be None"
        assert self.average_reward_steps > 1, "Reward must be averaged on more than 1 episode"
        
        # validate the agent is ready for training
        assert self.agent.validate()

        
        # do the magic here, i.e., the main training loop
        rewards = deque(maxlen=self.average_reward_steps)
        average_reward = 0

        for i in range(self.num_episodes): 
            # 1 and 2- reset the environement and get initial state
            state = self.env.reset()
            # 3 get first action
            action = self.agent.get_action(state)
            # 4 - iterate over the episode step unitl the agent moves to a terminal state or 
            # the episode ends 
            step = 0
            done = False
            episode_reward = 0
            while not done:
                # call step function in the environement
                state, reward, done, others = self.env.setp(action)
                episode_reward += reward

                # Call learn function in the agent. To learn for the last experience
                self.agent.learn(step, state, reward, done, others)
                action = self.agent.get_action(state)

            rewards.append(episode_reward)
            average_reward = sum(rewards)/self.average_reward_steps


            






        




