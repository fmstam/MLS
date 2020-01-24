# here we create a sample scenario and feed it to the rl_run file
# scenario_ac.py
# an actor-critic agent scenario
# we need to define 5 components for a standard DRL scenario/system:
# state and action spaces, environement, neural networks, replay memory, and agent.
# since this is an A2C agent, we do not use a replay memory
# Then we need to define some training paramaters to be used in the training manager
# that is it.
# use this template file to create any scenario


from MLS.examples.all_one.EnvEmptySlot import EnvEmptySlot
from MLS.torchDRL.DNN import ACDNN
from MLS.torchDRL.ACAgent import ACAgent


import numpy as np
###### main components of the scenario go here ######




# state and action
state_size = 30
action_space = [i for i in range(state_size)]
state_type = np.int16
action_type = np.int16

####### training options to be used by the training manager #######
num_episodes = 5000
episode_length = 2 * state_size
log_file = 'scenario_name_log_file.txt'


# envirnoment
env = EnvEmptySlot(state_size=state_size, action_space=action_space)

# neural nets
device = 'gpu'
hidden_layers_sizes = [64, 64, 64]
lr = 0.0001

actor_critic = ACDNN(input_shape=state_size,
            a_output_shape=len(action_space), # probs
            c_output_shape=1, # value estimation
            hidden_layers_sizes=hidden_layers_sizes,
            lr=lr,
            device=device)




# replay memory

# agent
discount_factor = 0.99
entropy_factor = 0.01

agent = ACAgent(state_size=state_size, 
                 action_space=action_space,
                 actor_critic=actor_critic,
                 discount_factor=discount_factor,
                 entropy_factor=entropy_factor,
                 episode_length=episode_length 
                 )


