""" 
    Implementaion of Deep Q network algorithm
    This class implements both DQN and Double DQN
     since they difference is just the calcualtion of the traget in the calculation of
    the mean square Bellman error.
    In addition, this class can easily carry out the Deuling DQN. What we need to change 
    is the DNN, in another class, see class DQNDNN in DNN.py.
"""

from MLS.ceot_drl.core.AbstractAgent import AbstractAgent
from MLS.ceot_drl.core.DNN import DQNDNN
from MLS.ceot_drl.core.utl.ReplayMemory import ReplayMemory

import numpy as np

class DQNAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space,
                 critic:DQNDNN, # Online critic
                 target_critic:DQNDNN,
                 replay_memory:ReplayMemory,
                 use_double=False, # true if we are going to use the DDQN algorithm instead 
                 use_smoothing=False, # true to use the Polyak averaging: weights = weights * \beta + (1 - \beta) new_weights
                 epsilon=0.99,
                 delta_epsilon=1e-4, # epsilon decay
                 min_epsilon=0.01,
                 discount_factor=0.99, # gamma
                 smoothing_frequency=20,
                 smoothing_factor=1e-2): 

        super(DQNAgent, self).__init__(critic=critic,
                                       actor=None, # DQN has no actor
                                       state_size=state_size,
                                       action_space=action_space,
                                       replay_memory=replay_memory)
        
        self.target_critic = target_critic
        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon
        self.min_epsilon = min_epsilon
        self.discount_factor = discount_factor
        self.use_smoothing = use_smoothing
        self.smoothing_factor = smoothing_factor
        self.smoothing_frequency = smoothing_frequency
        self.use_double = use_double


    def validate(self, parameter_list):
        super.validate()

    def get_action(self, state):
        """ Apply the forward on the critic network and return the action.
        Please note this is different from the get_action_epsilon_greedy function,
         which follows eps-greedy algorithm. 
        return:
        action -- an integer value belongs to action space
        """
        return np.argmax(self.critic.predict(state))
    
    def get_policy_action(self, state):
        return self.get_action_epsilon_greedy(state)

    def get_action_epsilon_greedy(self, state):
        """ epsilon-greedy algorithm.
        """
        if self.epsilon <= np.random.rand():
            return self.get_action(state)
        
        return np.random.choice(self.action_space)
    
    def anneal(self):
        """ Anneal epsilon, i.e cool down the exploration
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon - self.epsilon * self.delta_epsilon
        else:
            self.epsilon = self.min_epsilon
    

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

        # algorithm steps

        
        # 1- store the experience into the memory
        self.replay_memory.remember(state, state_, reward, action, done)

        # train only when there is enough data in the reply memory
        if total_steps > self.replay_memory.batch_size:
            # 2- sample random mini_batch
            state, state_, reward, action, done = self.replay_memory.sample() 

            # 3- core learning steps. 
            Q_state = self.critic.predict(state) # current state prediction from online network Q(s, a; \theta) 
            Q_state_ = self.target_critic.predict(state_) # next state prediction from the target critic Q(s', a; \theta^-)

            # we can replace the following loop by a single line via broadcasting, 
            # but I prefere it be explicit for equation-code readability
            for i in range(self.mini_batch_size):
                # we have two cases, 1) state_ is terminal 2) state_ is not terminal
                # if state_ is terminal 
                if done[i] == 1:
                    Q_state[i, action[i]] = reward[i]
                else: 
                    # here we use the DQN variant the agent will follow, 
                    # not the best approach but is informative
                    if self.use_double: # use Double DQN
                        act = np.argmax(Q_state_[i,:])
                        Q_state[i, action[i]] = reward[i] + self.discount_factor * Q_state[i, act]
                    else: # traditional DQN
                        Q_state[i, action[i]] = reward[i] + self.discount_factor *  np.max(Q_state_[i,:])

                # do fitting again
            self.critic.fit(state, Q_state)
            
            # Update the target critic weights accroding to smoothing_frequency
            if total_steps % self.smoothing_frequency == 0:
                self.target_critic.update_weights(self.critic, smoothing=self.use_smoothing, smoothing_factor=self.smoothing_factor)
            
            # anneal epsilon
            self.anneal()     