#!/usr/bin/env python
""" 
    Implementaion of Actor-critic algorithm
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

from MLS.torchDRL.AbstractAgent import AbstractAgent
from MLS.torchDRL.DNN import DNNArch
from MLS.torchDRL.utl.ReplayMemory import ReplayMemory

import numpy as np


class ACAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space,
                 actor_critic:DNNArch,
                 epsiode_length,
                 discount_factor=0.99,
                 entropy_factor=0.01):

        super(ACAgent, self).__init__()

        self.actor_critic = actor_critic
        self.discount_factor = discount_factor
        self.entropy_factor = entropy_factor
        self.epsiode_length = epsiode_length

        # episode rollout storage
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.entropy = 0

    def get_policy_action(self, state):
        # get action from the distribution
        # store the outcome into the rolout storage variables

        # do a forward pass on the ac network
        v, probs = self.actor_critic.predict(state)
        # get the action from the probs
        action = np.random.choice(len(self.actor_critic), p=probs)
        # calculate log(pi(action|state))
        log_prob = np.log(probs[action])
        # entropy
        entropy = np.sum(probs * np.log(probs))

        # store them, 
        # the reward will be added in the learn function.
        # the reason is that the reward is not avialable now
        self.values.append(v)
        self.log_probs.append(log_prob)
        self.entropy += entropy
    
    def learn(self, *args):
        """ The actual algorithm of DQN goes here
        
        Keyword arguments:
        *arg -- an experience sequence sent from the episode manageer. It should be unpacked to with this order: 
            step: the step in the episode
            state: the current state 
            state_: next state
            reward: the reward 
            done: if it was a terminal state
            extras: any application dependant observations, usually it is None and is ignored
        """

        # unpack args
        total_steps, episode_step, state, state_, reward, action, done, _ = args 

        # we check if we are do, so we do a learning step,
        # otherwise, just store the reward and go on

        if done:
            # calculate the discrounted reward
            discounted_rewards = np.zeros_like(self.values)
            for i in range(rang(len(self.rewards))):

        else:
            self.rewards.append(reward)












        
