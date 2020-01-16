# here we create a sample scenario and feed it to the rl_run file
# scenario.py

from MLS.examples.all_one.EnvEmptySlot import EnvEmptySlot
from MLS.torchDRL.DNN import DNN
from MLS.torchDRL.DQNAgent import DQNAgent 


# state and action
state_size = 5
action_space = [i for i in range(state_size)]

# envirnoment
env = EnvEmptySlot(state_size=state_size, action_space=action_space)

# neural nets
critic = DNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=[16],
            device='cpu')

target_critic = DNN(input_shape=state_size,
            output_shape=len(action_space),
            hidden_layers_sizes=[16],
            device='cpu')

# agent