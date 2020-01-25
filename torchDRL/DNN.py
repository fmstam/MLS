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
   
import torch   
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np

class DNNArch(nn.Module):
    def __init__(self,
                 input_shape, 
                 output_shape, 
                 hidden_layers_sizes=[16, 16], 
                 device='cpu', 
                 lr=1e-4):
        
        super(DNNArch, self).__init__()
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

        # output layer   
        actions = F.softmax(self.layers[-1](x))

        return actions # actions
    

    def summary(self):
        print(self)


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
             
if __name__ == "__main__":
    # execute only if run as a script
    pass
   
    
    

        
        







