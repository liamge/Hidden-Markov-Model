import nltk

class HMM:

    def __init__(self, obs_states):
        # Observation and State tuple (obs, state)
        self.obs_states = obs_states
        # Generate both list of all observations and states
        self.all_states = [s for (o, s) in self.obs_states]
        self.all_observations = [o for (o, s) in self.obs_states]
        # Generate set of observations and states
        self.states = set(self.all_states)
        self.observations = set(self.all_observations)
        # Emission Matrix
        self.emissionMatrix = {}
        # Transition Matrix
        self.transitionMatrix = {}

    def initTM(self):
        # Initializes the transition matrix
        stateBigrams = nltk.bigrams(s for (o, s) in self.obs_states)
        stateBigrams = [b for b in stateBigrams]
        cfd = nltk.ConditionalFreqDist(stateBigrams)
        fd = nltk.FreqDist(self.all_states)
        # Rows in matrix
        for o in self.states:
            self.transitionMatrix[o] = {}
            # Columns in matrix
            for o2 in self.states:
                # Equation is cfd[prevObs][nextObs]/fd[prevObs]
                self.transitionMatrix[o][o2] = cfd[o][o2]/fd[o]

    def transition_p(self,state1,state2):
        # Finds probability of state 2 occurring after state 1
        return self.transitionMatrix[state1][state2]

    def initEM(self):
        cfd = nltk.ConditionalFreqDist(self.obs_states)
        fd = nltk.FreqDist(self.all_states)
        for s in self.states:
            self.emissionMatrix[s] = {}
            for o in self.observations:
                # Equation is cfd[observation][state] / fd[state]
                self.emissionMatrix[s][o] = cfd[o][s]/fd[s]

    def emission_p(self,observation,state):
        # Finds the probability of an observation appearing as a state
        return self.emissionMatrix[state][observation]
