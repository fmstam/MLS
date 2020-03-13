""" 
    A simple environement.
    Given a binary array of length n, if the agent select empty slots then it receives 1,
     or the episode ends if it selects a non-empty slot
"""


from MLS.ceot_drl.core.AbstractEnvironment import AbstractEnvironement
import numpy as np

class EnvEmptySlot_AC(AbstractEnvironement):
    def __init__(self,
                 state_size = 5,
                 action_space = [i for i in range(5)]):
        super(EnvEmptySlot_AC, self).__init__(state_size, action_space)
        
        self.play_ground = np.zeros(self.state_size)
        self.action_space = [i for i in range(self.state_size)]

        
    def step(self, action):
        done = 0
        reward = 0
        extra_signals = []
        if self.play_ground[action] == 1:
            done = 1
        else:
            reward = 1
            self.play_ground[action] = 1
        
        if (self.play_ground == 1).all(): # board is full
            done = 1
            
        state_ = self.play_ground.copy()

        return state_, reward, done, extra_signals

    def reset(self):
        self.play_ground = np.zeros(self.state_size)
        self.play_ground[np.random.randint(low=0, high=self.state_size)] = 1

        return self.play_ground.copy()