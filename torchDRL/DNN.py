#!/usr/bin/env python
""" 
    Deep Neural Network class using torch.nn
"""
__author__ = "AL-Tam Faroq"
__copyright__ = "Copyright 2020, UALG"
__credits__ = ["Faroq AL-Tam"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Faroq AL-Tam"
__email__ = "ftam@ualg.pt"
__status__ = "Production"
   

# torch stuff
import torch   
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# numpy
import numpy as np




class GenericDNNArch(nn.Module):
    def __init__(self,
                 input_shape, 
                 output_shape, 
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        
        super(GenericDNNArch, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers_sizes = hidden_layers_sizes

        self.device = 'cpu' # default is the cpu
        if device is 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            
        
        # implement the actual neural network
        # first layer
        self.layers = nn.ModuleList([nn.Linear(input_shape, hidden_layers_sizes[0])]) 
        # loop over the network depth
        for i in range(1, len(hidden_layers_sizes)): 
            self.layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
        # last layer
        self.layers.append(nn.Linear(hidden_layers_sizes[-1], output_shape))

        # optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fun = nn.MSELoss(reduction='sum')

        # put the model in the self.device
        self.to(self.device)


    def forward(self, observation):
        
        x = torch.Tensor(observation).to(self.device)

        # forward loop
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))

        return x # actions
    
    def summary(self):
        print(self)

class DNNArch(GenericDNNArch):
    def __init__(self,
                 input_shape, 
                 output_shape, 
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):

        super(DNNArch, self).__init__(input_shape=input_shape,
                                  output_shape=output_shape,
                                  hidden_layers_sizes=hidden_layers_sizes,
                                  device=device,
                                  lr=lr)

    # we keep default loss and optimizer here

    def forward(self, x):
    # output layer   
        x = super.forward(x)
        actions = F.softmax(self.layers[-1](x))
        return actions
    
class DNNDeulingArch(nn.Module):
    def __init__(self,
                 input_shape, 
                 output_shape, 
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        
        super(DNNDeulingArch, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers_sizes = hidden_layers_sizes

        self.device = 'cpu' # default is the cpu
        if device is 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            
        
        # implement the actual neural network
        # first layer
        self.layers = nn.ModuleList([nn.Linear(input_shape, hidden_layers_sizes[0])]) 
        # loop over the network depth
        for i in range(1, len(hidden_layers_sizes)): 
            self.layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))

        # value-advantage layer
        self.A_layer = nn.Linear(hidden_layers_sizes[-1], self.output_shape)
        self.V_layer = nn.Linear(hidden_layers_sizes[-1], 1)
        
    
        # optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fun = nn.MSELoss(reduction='sum')

        # put the model in the self.device
        self.to(self.device)


    def forward(self, observation):
        x = torch.Tensor(observation).to(self.device)

        # forward loop
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))

        # now let AV-layer takes care of x 
        x_A = self.A_layer(x)
        x_V = self.V_layer(x)
        # combine both A and V layers
        # subtracted mean formula
        actions = x_V + (x_A - x_A.mean())

        return actions # actions
    

    def summary(self):
        print(self)

# combined actor critic DNN
class DNNACArch(nn.Module):
    def __init__(self,
                 input_shape, 
                 a_output_shape, 
                 c_output_shape=1,
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        super(DNNACArch, self).__init__()

        self.input_shape = input_shape
        self.a_output_shape = a_output_shape
        self.c_output_shape = c_output_shape
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = 'cpu'
        self.lr = lr

        if device is 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
        
        ## models
        # actor network
        # first layer
        self.a_layers = nn.ModuleList([nn.Linear(self.input_shape, self.hidden_layers_sizes[0])])
        # other layers
        for i in range(1,len(self.hidden_layers_sizes)):
            self.a_layers.append(nn.Linear(self.hidden_layers_sizes[i-1], self.hidden_layers_sizes[i]))
        # output layer
        self.a_layers.append(nn.Linear(self.hidden_layers_sizes[-1], self.a_output_shape))

        # critic network
        self.c_layers = nn.ModuleList([nn.Linear(self.input_shape, self.hidden_layers_sizes[0])])
        # hidden layers
        self.c_layers.extend([nn.Linear(self.hidden_layers_sizes[i-1], self.hidden_layers_sizes[i]) for i in range(1, len(self.hidden_layers_sizes))])
        # output layer
        self.c_layers.append(nn.Linear(self.hidden_layers_sizes[-1], self.c_output_shape))


        # opitmizer, the loss is defined seperately 
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss(reduction='sum')

        self.to(self.device)
        

    def forward(self, observation):

        x = torch.Tensor(observation).to(self.device)
        v = torch.Tensor(observation).to(self.device)
        for i in range(len(self.hidden_layers_sizes)):
            x = F.tanh(self.a_layers[i](x))
            v = F.relu(self.c_layers[i](v))

        # generate probabilities from the last layer
        probs = F.softmax(self.a_layers[-1](x))
        # generate value from the last layer
        v = self.c_layers[-1](v)[0]

        # return value estimate from the critic and probabilty distribution from the actor
        return v, probs

####################### seperated actor critic classes
# actor DNN
class Actor(GenericDNNArch):
    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_layers_sizes,
                 lr=1e-4,
                 device='cpu'
                ):
        super(Actor, self).__init__(input_shape=input_shape,
                                  output_shape=output_shape,
                                  hidden_layers_sizes=hidden_layers_sizes,
                                  device=device,
                                  lr=lr)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = super.forward(x)
        # we need to squash the output to -1 and +1 domain
        # when testing the agent we can relax the output to fit
        # the environement action space, see https://github.com/openai/gym/blob/master/gym/core.py
        # for action wraping

        actions = nn.functional.tanh(x)
        return actions
# critic DNN
class Critic(GenericDNNArch):
    def __init__(self,
                input_shape,
                output_shape,
                hidden_layers_sizes,
                lr=1e-4,
                device='cpu'
            ):
        super(Critic, self).__init__(input_shape=input_shape,
                                  output_shape=output_shape,
                                  hidden_layers_sizes=hidden_layers_sizes,
                                  device=device,
                                  lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    # eveything else is the same for the generic


##################################### warpers #########################
# neural net class for DQN models
class DQNDNN:
    def __init__(self,
                 input_shape, 
                 output_shape, 
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = device
        self.lr = lr

        # change only this line to include any different architecture in pytorch
        # we have two architecture DNNArch and DNNDeulingArch.
        self.model = DNNDeulingArch(input_shape=self.input_shape,
                             output_shape=self.output_shape, 
                             hidden_layers_sizes=self.hidden_layers_sizes, 
                             device=self.device,
                             lr=self.lr)

    def summary(self):
        self.model.summary()

    def predict(self, source):
        return self.model(source).detach().cpu().numpy()
    

    def fit(self, source, y, epochs=1):
        y_pred = self.model(source)
        loss = self.model.loss_fun(y_pred, torch.Tensor(y).to(self.model.device))
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def to_model_device(self, x):
        """ place a variable into the same device as the model    

        keyword arguments:
        x -- a variable

        return:
        A platform dependant variable placed in the same device as the model
        """

        x = torch.Tensor(x)
        x.to(self.model.device)
        return x

    def update_weights(self, dnn, smoothing=False, smoothing_factor=1e-3):
        """ Copy weights from another DNN
         
        keyword arguments:
        dnn -- another DNN must use the same lib, e.g., 
        smoothing -- if true the the weights are updated with a smoothing  factor
        smoothing_factor -- used if smoothing is true
        """
        if not smoothing:
            self.model.load_state_dict(dnn.model.state_dict())
        else:
            for param1, param2 in zip(self.model.parameters(), dnn.model.parameters()):
                param1.data.copy_(smoothing_factor * param1 + (1 - smoothing_factor) * param2)

# neual net class for Actor-critic agents
class ACDNN:
    def __init__(self,
                 input_shape, 
                 a_output_shape, 
                 c_output_shape=1,
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        self.input_shape = input_shape
        self.a_output_shape = a_output_shape
        self.c_output_shape = c_output_shape
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = device
        self.lr = lr
        
        self.model = DNNACArch(input_shape=self.input_shape,
                               a_output_shape=self.a_output_shape,
                               c_output_shape=self.c_output_shape,
                               hidden_layers_sizes=self.hidden_layers_sizes, 
                               device=self.device, 
                               lr=self.lr)


    def predict(self, source):
        v, probs = self.model(source)
        return v.detach().cpu().item(), probs.detach().cpu().numpy()

    def collect(self, source):
        v, probs = self.model(source)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return v, dist, action.cpu().detach().item(), log_probs, entropy

    
    def calc_loss(self,
                    discounted_r,
                    values,
                    log_probs,
                    entropy,
                    entropy_factor=0.01):
        """ Calculate the loss function and do the backward step

        keyword arguments:
        discounted_r -- the estimated Q in the Advantage equation: A_t(s, a) = r_{t+1} + gamma v_{t+1}(s) - v_t(s)
        values -- the esitmated values produced by the ciritic model
        log_probs -- the log of the distribution of the actions produced by the actor model
        entropy -- the entropy term which is used to encourage exploration. It is calcualted from probs
        entropy_factor -- is the contribution of the entropy term in the loss. Higher value means higher exploration.

        """

        discounted_r = torch.from_numpy(discounted_r).to(self.model.device)
        values = torch.stack(values).to(self.model.device)
        log_probs = torch.stack(log_probs).to(self.model.device)
        entropy = torch.stack(entropy).sum().to(self.model.device)

        # critic loss
        adv = discounted_r.detach() - values
        critic_loss = 0.5 * adv.pow(2).mean()

        # actor loss
        actor_loss = -(log_probs * adv.detach()).mean()

        loss = actor_loss + entropy_factor * entropy + critic_loss

        # reset grads
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

# DDGP DNN wrapper
class DDPGDNN:
    """ Impelement the computional steps of the DDPG algorithm.
    It contains everything related to DNN and their update to simplify the agent classes.
    """
    def __init__(self,
                 state_size,
                 action_size, # number of actions
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=[1e-4,1e-4]): # lr for actior and critic 
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.device = device
        self.lr = lr

        # actor 
        self.actor = Actor(input_shape=state_size,
                           output_shape=action_size,
                            hidden_layers_sizes=hidden_layers_sizes,
                            lr=lr[0],
                            device=device)
        self.actor_target = Actor(input_shape=state_size,
                           output_shape=action_size,
                            hidden_layers_sizes=hidden_layers_sizes,
                            lr=lr[0],
                            device=device)
        # critic
        self.critic = Critic(input_shape=state_size + action_size,
                            output_shape=action_size,
                            hidden_layers_sizes=hidden_layers_sizes,
                            lr=lr[1],
                            device=device)
        self.critic_target = Critic(input_shape=state_size + action_size,
                            output_shape=action_size,
                            hidden_layers_sizes=hidden_layers_sizes,
                            lr=lr[1],
                            device=device)

    def predict_actor(self, state):
        """ Predict output from the actor and return a numpy array
        """
        return self.actor(state).detach.cpu().numpy()

    def train_critic(self, states, actions, rewards, next_states, dones, discount_factor):

        """ Calculate the critic loss function and fit the state and apply one optimizer learning step.
        The model being used is the critic_target.
        
        keyword arguement:
        state -- current state numpy array sampled from the replymemory in the agent class
        actions -- actions mumpy array also sampled from the replaymemory in the agent class
        rewards -- rewards ....
        next_states -- next state ....
        dones -- dones ....
        discount_factor -- gamma in the equation
        """
        
        # calculate the loss:
        
        # Q values from the critic
        Q_critic = self.critic(states, actions)
        # actions from the actor
        actions_actor = self.actor_target(states)
        # Q values from the target critic using actions_actor
        Q_taget_critic = self.critic_target(next_states, actions_actor.detach())
         # we detach actions_actor to remove it from the computional graph of the target critic
        # calculate y 
        y = rewards + (1 - dones) * discount_factor * Q_taget_critic

        # loss function
        cirtic_loss = nn.MSELoss(Q_critic, y) # mean squared belman error (MSBE)

        # optimize
        self.critic.optimizer.zero_grad()
        cirtic_loss.backward()
        self.critic.optimizer.step()


    def train_actor(self, states):
        """ Train the actor using current states.
            We are trying to maximize the output of the Q_critic network at the action 
            selected by the actor network 
        """

        actor_loss = - self.critic(states, self.actor(states))

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
    
    def update_targets(self, smoothing=False, smoothing_factor=1e-3):

        """ Copy weights to target actor and critic
         
        keyword arguments:
        dnn -- another DNN must use the same lib, e.g., 
        smoothing -- if true the the weights are updated with a smoothing  factor
        smoothing_factor -- used if smoothing is true
        """
        if not smoothing:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            for param1, param2 in zip(self.critic_target.parameters(), self.critic.parameters()):
                param1.data.copy_(smoothing_factor * param2 + (1 - smoothing_factor) * param1)

            for param1, param2 in zip(self.actor_target.parameters(), self.actor.parameters()):
                param1.data.copy_(smoothing_factor * param2 + (1 - smoothing_factor) * param1)
        

                
if __name__ == "__main__":
    # execute only if run as a script
    pass
   
    
    

        
        







