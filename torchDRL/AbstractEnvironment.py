#!/usr/bin/env python
""" 
    An abstract envrionement class
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

 
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

