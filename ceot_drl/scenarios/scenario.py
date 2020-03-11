# here we create a sample scenario and feed it to the rl_run file
# scenario.py
# we need to define 5 components for a standard DRL scenario/system:
# state and action spaces, environement, neural networks, replay memory, and agent.
# Then we need to define some training paramaters to be used in the training manager
# that is it.
# use this template file to create any scenario


from MLS.ceot_drl.scenarios.EnvEmptySlot import EnvEmptySlot
from MLS.ceot_drl.core.DNN import DQNDNN
from MLS.ceot_drl.core.DQNAgent import DQNAgent 
from MLS.ceot_drl.core.utl.ReplayMemory import ReplayMemory

import numpy as np

title = 'Scenaro: Solving empty slot problem using DQN\n'
###### main components of the scenario go here ######
# state and action
state_size = 20
action_space = [i for i in range(state_size)]
state_type = np.int16
action_type = np.int16

# envirnoment
env = EnvEmptySlot(state_size=state_size, action_space=action_space)

# neural nets
device = 'gpu'
hidden_layers_sizes = [64, 64, 64]
lr = 0.0001

critic = DQNDNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=hidden_layers_sizes,
            lr=lr,
            device=device)

target_critic = DQNDNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=hidden_layers_sizes,
            device=device)

critic.summary()

# replay memory
replay_memory = ReplayMemory(state_type=state_type,
                             action_type=action_type,
                             state_size=state_size,
                             action_size=len(action_space))

# agent
delta_epsilon=1e-3
smoothing_frequency = 100
min_epsilon = 0.0005
use_double = True # Double DQN
agent = DQNAgent(state_size=state_size, 
                 action_space=action_space,
                 critic=critic,
                 target_critic=target_critic,
                 replay_memory=replay_memory,
                 delta_epsilon=delta_epsilon,
                 min_epsilon=min_epsilon,
                 smoothing_frequency = smoothing_frequency,
                 use_double=use_double)

####### training options to be used by the training manager #######
num_episodes = 2000
episode_length = 4 * state_size
log_file = 'scenario_name_log_file.txt'


