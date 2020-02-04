# Deep reinforcement learning package (MLS)
## Why MLS?
MLS is designed such that it abstracts almost everything. Therefore, reqardless of which platform you defined your neural nets, it still can run smoothly.

Current examples are implemented in **pytorch** but MLS should support any platform nicely. I am working on that part. I have just worked for two afternoons so far, I am planning to finish it soon, but you can contribute if you want.

The neural networks are implemented in pytroch and the package contains a Training Manager and is designed such that it reduces the overhead in the agents classes.

The package is composed of these main parts:
 #### 1- Core classes: 
  These are used to create replay memory, neural networks, agent algorithms, .... and so on. They are all abstract so that the y can be reshaped according to the required scenario
 #### 2 - Environement class:
 This class is abstract as well, and can be reshaped to fit any environment.
 #### 3 - Training Manager class:
 This class takes care of the training task, where it handels the training style.
 #### 4- scenario file:
 A scenario file is used to define all paramaters for the core and environmenet class. As well as the training options.
 
 #### 5- run file:
 The code in the main function is in principle the same for every RL scenario. However, this file main role is to read the scenario file and send it to the training manager to run it
 
## How to use it:
It is simple, see the example in `scenario.py`. You need to subclass the abstract environement class `AbstractEnvironment`, create neural networks, using any platform you like (torch, TF, Keras, and so on), and then create an agent. That is it.



## Implemented algorithms 
 ### 1- DQN (included)
 ### 2- Double DQN (included)
 ### 3- Deuling Double DQN (included)
 ### 4- Actor-Critic (A2C) with n-steps (included)
 ### 7*- Deep Deterministic Policy Gradient (DDPG) (included)
 ### 5 - DQN with Prioritized Experience Replay (in progress)
 ### 6 - DQN with LSTM (queued)
 ### 7 - TD3 (queued)

 
 

