from matplotlib import pyplot as plt
import torch

class Plot():
    def __init__(self, title, xlabel, ylabel, values, moving_avg_period = 0):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.values = values
        self.moving_avg_period = moving_avg_period

    def show(self):
        #below does not work on mac
        #plt.interactive(False)
        #plt.ioff()
        plt.figure(self.title)
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.plot(self.values)
		# plt.axes[] set_ylim (0.5)
        if self.moving_avg_period > 0:
            plt.plot(get_moving_average(self.moving_avg_period, self.values))

        plt.show()

    @staticmethod
    def get_moving_average(period, values):
        values = torch.tensor(values, dtype=torch.float32)
        if len(values) >= period:
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()