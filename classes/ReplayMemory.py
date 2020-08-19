import random

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    #store experiences as they occur
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            #push new experience to front of memory - overriding oldest experiences first
            self.memory[self.push_count % self.capacity] = experience 
        self.push_count += 1
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    #tell us if we can sample from memory - so if the batch size is 50 but memory is only at 20 values
    #this will return false
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size