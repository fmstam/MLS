#!/usr/bin/env python
""" 
    Implementaion of Deep Determinist Policy Gradient agent. 
    This class heavely uses the DDPGDNN in the DNN file. 
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
from MLS.torchDRL.utl.ReplayMemory import ReplayMemory
from MLS.torchDRL.utl.OUNoise import OUNoise 
from MLS.torchDRL.utl.ActionWrapper import ActionWrapper
from MLS.torchDRL.DNN import DDPGDNN as DDPGDNN # DNN stuff are handeled here


class DDPGAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space, # 2 X number of actions matrix (i.e., upper and lower bounds of each action)
                 neural_net_wrapper:DDPGDNN, # actor-critic network wrapper
                 replay_memory: ReplayMemory,
                 discount_factor=0.99,
                 use_smoothing=True, # true to use the Polyak averaging: weights = weights * \beta + (1 - \beta) new_weights
                 smoothing_frequency=1, 
                 smoothing_factor=1e-2,
                 mini_batch_size=128):

        super(DDPGAgent, self).__init__(state_size=state_size,
                                        action_space=action_space,
                                        replay_memory=replay_memory,
                                        mini_batch_size=mini_batch_size)

        self.nn_wrapper = neural_net_wrapper
        self.noise = OUNoise(action_dim=len(self.action_space), low=-1.0, high=1.0) # noise term
        self.action_wrapper = ActionWrapper(self.action_space) # action wrapper 
        
        self.discount_factor = discount_factor

        self.use_smoothing = use_smoothing
        self.smoothing_frequency = smoothing_frequency
        self.smoothing_factor = smoothing_factor
        
        self.step = 0 # current step in the episode used in noising the action,
        

    # the policy action
    def get_action(self, state):
        return self.nn_wrapper.predict_actor(state)

    def get_policy_action(self, state):
        """
        Get policy action, add noise and map it to the correct action space.
        """
        actions =  self.get_action(state) # get action from actor
        actions = self.noise.get_action(actions, self.step) # add noise term

        return actions # note we can have more than one action 

    

    def learn(self, *args):
        """ learn from a reply memory
        
        Keyword arguments:
        *arg -- an experience sequence sent from the episode manageer. 
        It should be unpacked to with this order: 

            step: the step in the episode
            state: the current state 
            state_: next state
            reward: the reward 
            done: if it was a terminal state
            extras: any application dependant observations, 
                usually it is None and is ignored
        """

        # unpack args
        total_steps, episode_step, state, state_, reward, action, done, _ = args 

        # algorithm steps

        self.step = episode_step # keep track of step to use it in action noising
        # 1- store the experience into the memory
        self.replay_memory.remember(state, state_, reward, action, done)

        # train only when there is enough data in the reply memory
        if total_steps > self.replay_memory.batch_size:
            # 2- sample random mini_batch
            state, state_, reward, action, done = self.replay_memory.sample() 
            self.nn_wrapper.train_critic(state, action, reward, state_, done, self.discount_factor)            
            self.nn_wrapper.train_actor(state)
            # Update the target critic weights accroding to smoothing_frequency
            if total_steps % self.smoothing_frequency == 0:
                self.nn_wrapper.update_targets(smoothing=self.use_smoothing, smoothing_factor=self.smoothing_factor)
      

    


