""" 
    An abstract envrionement class
"""
 
class AbstractEnvironement():
    def __init__(self,
                 state_size,
                 action_space):
        self.state_size = state_size
        self.action_space = action_space
    
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

