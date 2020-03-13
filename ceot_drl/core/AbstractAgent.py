""" 
    An abstract agent
"""


#from abc import ABC
from MLS.ceot_drl.core.utl.ReplayMemory import ReplayMemory

class AbstractAgent():
    def __init__(self,
                 state_size, 
                 action_space,
                 actor=None, 
                 critic=None, 
                 replay_memory=None, 
                 mini_batch_size=64):
        self.actor = actor
        self.critic = critic
        self.replay_memory = replay_memory
        self.state_size = state_size
        self.action_space = action_space
        self.mini_batch_size = mini_batch_size

    def validate(self):
        return True
            # TO BE FINISHED LATER
            #assert self.state_size is not None, 'state shape can not be None'
            #assert self.state_size < 1
    
    def learn(self, *args):
        raise NotImplementedError

    def get_policy_action(self, state):
        raise NotImplementedError
    
    def get_action(self, state):
        raise NotImplementedError

    def get_critic(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError