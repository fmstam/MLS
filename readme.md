# Deep reinforcement learning package in pytorch
The neural networks are implemented in pytroch and the package contains a Training Manager and is designed such that it reduces the overhead in the agents classes.

The package is composed of these main parts:
 #### 1- core classes: 
  These are used to create replay memory, neural networks, agent algorithms, .... and so on. They are all abstract so that the y can be reshaped according to the required scenario
 #### 2 - Environement class:
 This class is abstract as well, and can be reshaped to fit any environment.
 #### 3 - Training Manager classL
 This class takes care of the training task, where it handels the training style.
 #### 4- scenario file:
 A scenario file is used to define all paramaters for the core and environmenet class. As well as the training options.
 
 #### 5- run file:
 The code in the main function is in principle the same for every RL scenario. However, this file main role is to read the scenario file and send it to the training manager to run it

## Implemented algorithms 
 ### 1- DQN 
 ### 2- DDQN
 ### 3- Actor-Critic (AC)
 ### 4- Policy Gradient
 ### 5- Deep Deterministic Policy Gradient

