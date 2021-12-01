import matplotlib.pyplot as plt
import numpy as np

# prepare plot before starting
def prepare_2Dplots(dataset):
    ds = np.array(dataset)
    # plt.style.use('seaborn-whitegrid')
    ax = plt.axes()
    ax.grid(alpha=0.3, zorder=0)
    plt.xlabel("X")
    plt.ylabel("Y")
    line = [None, None]
    line[0], = ax.plot(ds[:, 0], ds[:, 1], 'k-', linewidth=0.8, label='Targeted', zorder=2) # 'b-' = blue line    
    line[1], = ax.plot(ds[:, 0], ds[:, 1], 'rD', markersize=6, fillstyle='none', markeredgecolor='r', markeredgewidth=0.8, label='Generated', zorder=3) # 'r-' = red line
    plt.ion() # interactive mode for plot
    return ax, line

# update plot
def plot2D(ax, line, new_dataset, last = False):
    nds = np.array(new_dataset)
    line[1].set_xdata(nds[:, 0])
    line[1].set_ydata(nds[:, 1])
    plt.draw()  
    plt.pause(0.01)
    if last:
        plt.ioff()
        # plt.legend([line[0], line[1]], ['label1', 'label2'], frameon=1)
        plt.legend(frameon=1)
        plt.show()

# prepare plot before starting
def prepare_3Dplots(dataset):
    ds = np.array(dataset)
    # plt.style.use('seaborn-whitegrid')
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    line = [None, None]
    line[0], = ax.plot3D(ds[:, 0], ds[:, 1], ds[:, 2], 'k-', linewidth=0.8, label='Targeted') # 'b-' = blue line    
    line[1], = ax.plot3D(ds[:, 0], ds[:, 1], ds[:, 2], 'rD', markersize=6, fillstyle='none', markeredgecolor='r', markeredgewidth=0.8, label='Generated') # 'r-' = red line
    plt.ion() # interactive mode for plot
    return ax, line

# update plot
def plot3D(ax, line, new_dataset, last = False):
    nds = np.array(new_dataset)
    line[1].set_xdata(nds[:, 0])
    line[1].set_ydata(nds[:, 1])
    line[1].set_3d_properties(nds[:, 2])
    plt.draw()  
    plt.pause(0.01)
    if last:
        plt.ioff()
        # plt.legend(frameon=1)
        plt.show()