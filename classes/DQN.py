import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, model, img_height, img_width):
        super().__init__()

        if(model.upper() == "LINEAR"):
             self.fc1, self.fc2, self.out = self.linear_network(img_height, img_width)
        elif(model.upper() == "CNN"):
            print("CNN")
       
    def linear_network(self, img_height, img_width):
        fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)
        fc2 = nn.Linear(in_features=24,  out_features=32)
        out = nn.Linear(in_features=32, out_features=2)

        return fc1, fc2, out
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t)) 
        t = F.relu(self.fc2(t))
        return t
    