# here we create a sample scenario and feed it to the rl_run file
# scenario_ac.py
# an actor-critic agent scenario
# we need to define 5 components for a standard DRL scenario/system:
# state and action spaces, environement, neural networks, replay memory, and agent.
# since this is an A2C agent, we do not use a replay memory
# Then we need to define some training paramaters to be used in the training manager
# that is it.
# use this template file to create any scenario


from MLS.ceot_drl.scenarios.EnvEmptySlot_AC import EnvEmptySlot_AC
from MLS.ceot_drl.core.DNN import ACDNN
from MLS.ceot_drl.core.ACAgent import ACAgent
import gym


import numpy as np

title = 'Scenario: CartePole-v1 using A2C n-step algorithm'
###### main components of the scenario go here ######
# env, state and action

# we can use gym environment 
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_space = [i for i in range(env.action_space.n)]

# or our simplified environement class
# env = EnvEmptySlot_AC(state_size=15)
# state_size = env.state_size
# action_space = env.action_space

state_type = np.int16
action_type = np.int16

####### training options to be used by the training manager #######
num_episodes = 500
episode_length = 500
log_file = 'scenario_name_log_file.txt'



# neural nets
device = 'gpu'
hidden_layers_sizes = [128, 128]
lr = 0.001

actor_critic = ACDNN(input_shape=state_size,
            a_output_shape=len(action_space), # probs
            c_output_shape=1, # value estimation
            hidden_layers_sizes=hidden_layers_sizes,
            lr=lr,
            device=device)




# replay memory

# agent
discount_factor = 0.99
entropy_factor = 0.0001

agent = ACAgent(state_size=state_size, 
                 action_space=action_space,
                 actor_critic=actor_critic,
                 discount_factor=discount_factor,
                 entropy_factor=entropy_factor,
                 episode_length=episode_length 
                 )



