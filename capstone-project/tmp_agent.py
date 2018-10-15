class DQLAgent(object):
    def __init__(self, states_sz, actions_sz):
        """ Initialise the Agent """
        pass

    def _build_model(self):
        """ Build Neural Net for Deep-Q-Learning """
        pass

    def remember(self, state, action, reward, next_state, done):
        """ Implement experiece replay dataset """
        pass

    def action(self, state):
        """ Return an action from current state """
        pass

    def replay(self, batch_sz):
        """ learn using replay experiece """
        pass

    def load(self, name):
        """ load weights """
        pass

    def save(self, name):
        """save weights """
        pass
