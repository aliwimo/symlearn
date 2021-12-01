import matplotlib.pyplot as plt
import numpy as np


VAR_NUM = 2
DOMAIN_X = [0, 1]
DOMAIN_Y = [0, 1]
POINT_NUM = 50

def sin(x): return np.sin(x)
def cos(x): return np.cos(x)
def exp(x): return np.exp(x)
def log(x):
    if x == 0: return 0
    else: return np.log(np.abs(x))


def target(x, y = 0):
    # return x*x*x*x + x*x*x + x*x + x
    # return x**5 + x**4 + x**3 + x**2 + x
    # return np.sin(x) + np.sin(x + (x**2))
    # return np.sin(x**2) * np.cos(x) - 1 
    # return np.log(x + 1) + np.log((x**2) + 1)
    # return np.sqrt(x)
    return np.sin(x) + np.sin(y ** 2)
    # return 2 * np.sin(x) * np.cos(y)

def generated(x0, x1=0):
    # return ((x)-((y)-(sin((x)+(y)))))
    return 0.38751683894644706+0.4351101981651027*log((((x0 * x1) * exp(x0)) + ((((x0 * x1) * exp((x0 * x1))) + ((((((x0 * x1) * exp(x0)) + (x1 * x1)) * x1) * exp(x0)) + ((((x0 * x1) * exp(x0)) + (x1 * x1)) * x1))) + (((x0 * x1) + (x1 * x1)) * x1))))

def orginal_dataset():
    dataset = []
    points_x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], POINT_NUM)
    if VAR_NUM == 1:

        for i in range(POINT_NUM):
            dataset.append([points_x[i], target(points_x[i])])
    else:
        points_y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], POINT_NUM)
        for i in range(POINT_NUM):
            dataset.append([points_x[i], points_y[i], target(points_x[i], points_y[i])])
    return dataset

def computed_dataset():
    dataset = []
    points_x = np.linspace(DOMAIN_X[0], DOMAIN_X[1], POINT_NUM)
    if VAR_NUM == 1:

        for i in range(POINT_NUM):
            dataset.append([points_x[i], generated(points_x[i])])
    else:
        points_y = np.linspace(DOMAIN_Y[0], DOMAIN_Y[1], POINT_NUM)
        for i in range(POINT_NUM):
            dataset.append([points_x[i], points_y[i], generated(points_x[i], points_y[i])])
    return dataset

# prepare plot before starting
def prepare_2Dplots(dataset):
    ds = np.array(dataset)
    plt.style.use('seaborn-whitegrid')
    ax = plt.axes()
    line = [None, None]
    line[0], = ax.plot(ds[:, 0], ds[:, 1], 'k-') # 'b-' = blue line    
    # line[1], = ax.plot(ds[:, 0], ds[:, 1], 'r.') # 'r-' = red line
    line[1], = ax.plot(ds[:, 0], ds[:, 1], 'r.') # 'r-' = red line
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
        plt.show()


# prepare plot before starting
def prepare_3Dplots(dataset):
    ds = np.array(dataset)
    plt.style.use('seaborn-whitegrid')
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    line = [None, None]
    line[0], = ax.plot3D(ds[:, 0], ds[:, 1], ds[:, 2], 'b-') # 'b-' = blue line    
    # line[1], = ax.plot3D(ds[:, 0], ds[:, 1], ds[:, 2], 'r--') # 'r-' = red line
    line[1], = ax.plot3D(ds[:, 0], ds[:, 1], ds[:, 2], 'r--') # 'r-' = red line
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
        plt.show()


ds1 = orginal_dataset()
ds2 = computed_dataset()
if VAR_NUM == 1:
    ax, line = prepare_2Dplots(ds1)
    plot2D(ax, line, ds2, last = True)
else:
    ax, line = prepare_3Dplots(ds1)
    plot3D(ax, line, ds2, last = True)