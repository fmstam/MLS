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
from MLS.torchDRL.DNN import DDPGDNN as DDPGDNN # DNN stuff are handeled here


class DDPGAgent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_space, # 2 X number of actions matrix (i.e., upper and lower bounds of each action)
                 neural_net_wrapper:DDPGDNN, # actor-critic network wrapper
                 replay_memroy: ReplayMemory,
                 discount_factor=0.99,
                 smoothing_frequency=20, 
                 smoothing_factor=1e-3,
                 mini_batch_size=64):
        super(DDPGAgent, self).__init__(state_size=state_size,
                                        action_space=action_space,
                                        replay_memory=replay_memory,
                                        mini_batch_size=mini_batch_size)