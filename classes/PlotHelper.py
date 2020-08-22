import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class PlotHelper:
    def check_is_python(self, values, moving_avg_period):
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython: from IPython import display

        print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
        if is_ipython: display.clear_output(wait=True)
    #used to display performance on a chart
    def plot(self, values, moving_avg_period):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.yPlotlabel('Duration')
        plt.plot(values)
        moving_avg = self.get_moving_average(moving_avg_period, values)
        plt.plot(moving_avg)
        plt.pause(0.0001)
        self.check_is_python()
        
    def get_moving_average(self, period, values):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= period:
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()