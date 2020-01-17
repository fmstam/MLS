# here we create a sample scenario and feed it to the rl_run file
# scenario.py
# we need to define 5 components for a standard DRL scenario/system:
# state and action spaces, environement, neural networks, replay memory, and agent.
# Then we need to define some training paramaters to be used in the training manager
# that is it.
# use this template file to create any scenario


from MLS.examples.all_one.EnvEmptySlot import EnvEmptySlot
from MLS.torchDRL.DNN import DNN
from MLS.torchDRL.DQNAgent import DQNAgent 
from MLS.torchDRL.utl.ReplayMemory import ReplayMemory

import numpy as np
###### main component of the scenario go here ######



# state and action
state_size = 5
action_space = [i for i in range(state_size)]
state_type = np.int16
action_type = np.int16

# envirnoment
env = EnvEmptySlot(state_size=state_size, action_space=action_space)

# neural nets
device = 'cpu'
hidden_layers_sizes = [16, 16]

critic = DNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=hidden_layers_sizes,
            device=device)

target_critic = DNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=hidden_layers_sizes,
            device=device)

# replay memory
replay_memory = ReplayMemory(state_type=state_type,
                             action_type=action_type,
                             state_size=state_size,
                             action_size=len(action_space),)

# agent
agent = DQNAgent(state_size=state_size, 
                 action_space=action_space,
                 critic=critic,
                 target_critic=target_critic,
                 replay_memory=replay_memory,
                 smoothing_frequency = 30)

####### training options to be used by the training manager #######
num_episodes = 1000
episode_length = 200
log_file = 'scenario_name_log_file.txt'


