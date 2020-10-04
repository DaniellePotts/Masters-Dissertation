import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging
logger = logging.getLogger()
logger.disabled = True

import minerl
import gym

import os

class MinecraftEnvironmentManager():
    def __init__(self, environment):
        self.environment = environment
        self.env = self.make_environment(self.environment)
    def set_data_path(self, data_path):
        os.environ['MINERL_DATA_ROOT']=data_path
    def change_environment(self, environment):
        return self.make_environment(environment)
    def make_environment(self, environment):
        return gym.make(self.environment)
    def load_data(self):
        return minerl.data.make(self.environment)
    def reset(self):
        self.env.reset()
    def get_action_sample(self):
        return self.env.action_space.sample()

