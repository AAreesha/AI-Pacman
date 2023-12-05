class SarsaAgent(ReinforcementAgent):
    """
    SARSA Agent

    Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state-action pair
        """
        return self.QValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.
        If there are no legal actions, return 0.0.
        """
        qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not qvalues:
            return 0.0
        return max(qvalues)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.
        If there are no legal actions, return None.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        return max(legalActions, key=lambda action: self.getQValue(state, action))

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Update the Q-value based on the SARSA update rule.
        """
        nextAction = self.getAction(nextState)
        oldQValue = self.getQValue(state, action)
        nextQValue = self.getQValue(nextState, nextAction) if nextAction is not None else 0.0

        sample = reward + self.discount * nextQValue
        self.QValues[(state, action)] = (1 - self.alpha) * oldQValue + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
