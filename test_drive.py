# test_drive.py
# I use this file to test things, 


import torch 


class Test(torch.nn.ModuleList):
    def __init__(self):
        super(Test, self).__init__()
        self.fc1 = torch.nn.Linear(2, 1)
    
    def forward(self, x):
        obs = torch.Tensor(x)
        y =  torch.nn.functional.relu(self.fc1(obs))
        return y

if __name__ is '__main__':
    t = Test()
    t([10, 20])


