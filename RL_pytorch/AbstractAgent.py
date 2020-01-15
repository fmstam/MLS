#!/usr/bin/env python
""" 
    An abstract agent
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"

from abc import ABC

class AbstractAgent():
    def __init__(self, actor=None, critic=None, replay_memory=None, state_shape=None, action_shape=None ):
        self.actor = actor
        self.critic = critic
        self.replay_memory = replay_memory
        self.state_shape = state_shape
        self.action_shape = action_shape
    
    def train(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_critic(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError


