#!/usr/bin/env python
""" 
Frame-based equipement simulation of MAC

"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"


import simpy


FRAME_LENGTH = 10000 # 10 ms
CCA_TIME = 20        # 20 us
IDLE_TIME = 50       # 5% FRAME_LENGTH

class Node:
    def __init__(self,
                 name,
                 env,
                 channel):
        self.name = name
        self.env = env
        self.channel = channel

    def join(self):
        # join the node pool
        self.action = env.process(self.access_channel())

    def access_channel(self):
        # try to access the channel

    


if __name__ is '__main__':
    main()