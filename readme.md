
**CEOT-DRL** is deep reinforcement learning (DRL) package mainly used in the Centro de Electrónica, Optoelectronica e Telecomunicações (CEOT) research center for DRL research on networking and systems. **CEOT-DRL** is designed such that it abstracts almost everything. Therefore, regardless of which computational graph platform is used to define the neural nets, it still can run smoothly. 

Published version of **CEOT-DRL** 0.1 supports only common DRL algorithms, but our own models/algorithms will be published here after the papers gets published. Currently supported algorithms:

## Implemented algorithms 
 ### 1- DQN (included)
 ### 2- Double DQN (included)
 ### 3- Deuling Double DQN (included)
 ### 4- Actor-Critic (A2C) with n-steps (included)
 ### 7*- Deep Deterministic Policy Gradient (DDPG) (included)
 ### 5 - DQN with Prioritized Experience Replay (in progress)
 ### 6 - DQN with LSTM (queued)
 ### 8 - TD3 (queued)


Current examples of **CEOT-DRL** are implemented in **pytorch** but it should support any platform. The reason is that, the agents are implemented in a computational graph lib-agnostic way.


##**CEOT-DRL** is composed of these main parts:
 
 #### 1- scenario file:
 A scenario file is used to define all parameters for the core and environment class. As well as the training options.
 It represents the problem, we want to solve, we feed it to a training manager, which run it and produce the results.
 
 #### 2- Core classes: 
  These are used to create replay memory, neural networks, agent algorithms, .... and so on. They are all abstract so that they can be reshaped according to the required scenario.
  Each agent has a neural net wrapper/architecture, which implements the neural net components separately. For example, the `DDPGAgent` class has a neural network wrapper `neural_net_wrapper:DDPGDNN` in the `DNN.py` file, which handles the neural network stuff independently from the `DDPGAgent` class

 #### 3 - Environment class:
 This class is abstract as well, and can be reshaped to fit any environment. In addition, to user-defined environments classes from openAI `gym` can be used. See scenario folder for example.

 #### 4 - Training Manager class:
 This class takes care of the training task, where it handles the training of the agents.

 #### 5- run file:
 This file main role is to read the scenario file and send it to the training manager to run it.
 
## How to use it:
As an example see `example.ipynb`. It has a scenario and uses the training manager to execute the scenario. For a scenario example see `scenario.py`. To create your own scenario, you need to subclass the abstract environment class `AbstractEnvironment`, create neural networks architecture, using any platform you like (torch, TF, Keras, and so on), and then create an agent(DQN, AC, ....).




 
 

