#!/usr/bin/env python
""" 
    Implementaion of Deep Q network algorithm
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
from DNN import DNN
from utl.ReplayMemory import ReplayMemory

import numpy as np

class DQNAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space,
                 critic:DNN, # Online critic
                 target_critic:DNN,
                 replay_memory:ReplayMemory,
                 use_double=False, # true if we are going to use the DDQN algorithm instead 
                 epsilon=0.99,
                 delta_epsilon=1e-4, # epsilon decay
                 min_epsilon=0.01,
                 discount_factor=0.99, # gamma
                 smoothing_frequency=20,
                 smoothing_factor=1e-3): 

        super(DQNAgent, self).__init__(critic=critic,
                                       state_size=state_size,
                                       action_size=len(action_space),
                                       replay_memory=replay_memory)
        
        self.target_critic = target_critic
        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = min_epsilon
        self.discount_factor = discount_factor
        self.smoothing_factor = smoothing_factor
        self.smoothing_frequency = smoothing_frequency


    def validate(self, parameter_list):
        super.validate()



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
        step, state, state_, reward, action, done, _ = args 

        # 1- store the experience into the memory
        self.replay_memory.reward(state, state_, reward, done)

        # 2- sample random mini_batch
        state, state_, reward, action, done = self.replay_memory.sample() 

        # 3- core learning steps. 
        
        Q_state = self.critic.predict(state) # current state prediction from online network Q(s, a; \theta) 
        Q_state_ = self.target_critic.predict(state_) # next state prediction from the target critic Q(s', a; \theta^-)

        # we can replace the following loop by a single line via broadcasting, 
        # but I prefere it be explicit here to describe the actual mathematical equation
        for i in range(self.mini_batch_size):
            # we have two cases, 1) state_ is terminal 2) state_ is not terminal
            # if state_ is terminal 
            if done[i] == 1:
                Q_state[i, action[i]] = reward[i]
            else: 
                Q_state[i, action[i]] = reward[i] + self.discount_factor *  np.max(Q_state_[i,:])
            # do fitting again
            self.critic.fit(state, Q_state)
        
        # Update the target critic weights accroding to smoothing_frequency
        if step % self.smoothing_frequency == 0:
            self.target_critic.copy_weights(self.critic, smoothing=True, smoothing_factor=self.smoothing_factor)
        
        # anneal epsilon
        self.anneal()


    
        



    def get_action(self, state):
        """ Apply the forward on the critic network and return the action.
        Please note this is different from the get_action_epsilon_greedy function,
         which follows eps-greedy algorithm. 
        return:
        action -- an integer value belongs to action space
        """
        return np.argmax(self.critic.forward(state).numpy())
    
    def get_action_epsilon_greedy(self, state):
        """ epsilon-greedy algorithm.
        """
        if self.epsilon <= np.random.rand():
            return self.get_action(state)
        
        return np.random.choice(self.action_space)
    
    def anneal(self):
        """ Anneal epsilon, i.e cooling down the exploration
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.delta_epsilon
        else:
            self.epsilon = self.min_epsilon


        
     
        
    





if __name__ == "__main__":
    # execute only if run as a script
   pass
    