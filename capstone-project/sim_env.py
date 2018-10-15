class SIMEnv(object):
    def __init__(self, proxy_table):
        """ Initilise the Environment with informations from proxy_table """
        pass

    def step(self, action):
        """ Take a action, calculate the reward and return these informations """
        pass

    def reset(self):
        """ Reset the Environment to the initial state """
        pass

    def seed(self, seed=None):
        """ Define the Environment seed """
        pass

    def _take_action(self, action):
        """ Select all informations from proxy table (Fat, Carbo, Protein, Energy) """
        pass

    def _get_reward(self):
        """ calculate the reward based on nutritional information """
        pass
