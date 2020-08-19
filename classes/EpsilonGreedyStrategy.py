import math

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay): #starting, ending a decaying values of epsilon
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)