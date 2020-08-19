import torch

class Utils:
    def __init__(self, experience):
        self.Experience = experience
    def extract_tensors(self, experiences):
        batch = self.Experience(*zip(*experiences))
        
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        
        return (t1,t2,t3,t4)