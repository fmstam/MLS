#!/usr/bin/env python
""" 
    DQN class
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

class DQNAgent(AbstractAgent):
    def __init__(self, 
                 state_shape=None,
                 action_shape=None,
                 critic=None, epsilo=0,
                 delta_epsilon=0,
                 replay_memory=None):

        AbstractAgent.__init__(self, critic=critic, replay_memory=replay_memory)

    def train(self):
        print('here goes the training')


def main():
    dqn = DQNAgent(critic=20)
    dqn.train()


if __name__ == "__main__":
    # execute only if run as a script
    main()
    