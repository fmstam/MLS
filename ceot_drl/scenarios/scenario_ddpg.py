# here we create a sample scenario and feed it to the rl_run file
# scenario_ddpg.py
# Here we assume that the action stpace is continouse with low and high limits



from MLS.ceot_drl.scenarios.EnvEmptySlot_AC import EnvEmptySlot_AC
from MLS.ceot_drl.core.DNN import DDPGDNN
from MLS.ceot_drl.core.DDPGAgent import DDPGAgent
from MLS.ceot_drl.core.utl.ReplayMemory import ReplayMemory
from MLS.ceot_drl.core.utl.ActionWrapper import NormalizedEnv

import gym
import numpy as np


###### main components of the scenario go here ######



# title 
title = ' Solving Pendulum-v0 problem using DDPG algorithm'
# env, state and action

# we can use gym environment 
env = NormalizedEnv(gym.make("Pendulum-v0"))
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_space = np.zeros((1,2))
action_space[0,0] = env.action_space.low
action_space[0,1] = env.action_space.high

# or our simplified environement class
# env = EnvEmptySlot_AC(state_size=15)
# state_size = env.state_size
# action_space = env.action_space



####### training options to be used by the training manager #######
num_episodes = 500
episode_length = 500
log_file = 'scenario_name_log_file.txt'



# neural nets wrapper
device = 'gpu'
hidden_layers_sizes = [256]
lr = [.0001, .001] # DDPGNN expect a list of lr one for actor and another for critic

ddpg_dnn_wrapper = DDPGDNN(state_size=state_size,
            action_size=len(action_space),
            hidden_layers_sizes=hidden_layers_sizes,
            device=device,
            lr=lr)




# replay memory
state_type = np.float32
action_type = np.float32
replay_memory = ReplayMemory(state_type=state_type,
                             action_type=action_type,
                             state_size=state_size,
                             action_size=len(action_space))

# agent
discount_factor = 0.99

agent = DDPGAgent(state_size=state_size, 
                 action_space=action_space,
                 neural_net_wrapper=ddpg_dnn_wrapper,
                 discount_factor=discount_factor,
                 replay_memory=replay_memory
                 )



