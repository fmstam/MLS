# https://github.com/openai/gym/blob/master/gym/core.py
class ActionWrapper:
    """ Action normalizer to the environmenet accepted range 
        It is required to map the output of the neural nets of the agents to an accepted range.
    """
    def __init__(self, action_space):
        self.action_space_lower = action_space[:, 0]
        self.action_space_upper = action_space[:, 1]
        

    def wrap_action(self, action):
        act_k = (self.action_space_upper - self.action_space_lower)/ 2.
        act_b = (self.action_space_upper + self.action_space_lower)/ 2.
        return act_k * action + act_b

    def unwrap_action(self, action):
        act_k_inv = 2./(self.action_space_upper - self.action_space_lower)
        act_b = (self.action_space_upper + self.action_space_lower)/ 2.
        return act_k_inv * (action - act_b)