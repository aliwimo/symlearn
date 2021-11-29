import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.random import check_random_state

def target_function(x):
    return x*x*x*x + x*x*x + x*x + x

np.random.seed(np.random.randint(200))
rng = check_random_state(0)
# X = rng.uniform(-1, 1, 20).reshape(20, 1)
X = np.random.uniform(-1, 1, 20).reshape(20, 1)
Y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]).reshape(20, 1)

dataset = []
x = np.linspace(-1, 1, 20)
for i in range(20):
    dataset.append([x[i], target_function(x[i])])

dataset = np.array(dataset)


ax = plt.axes()
ax.grid(alpha=0.3, zorder=0)
plt.xlabel("X")
plt.ylabel("Y")
line = [None, None]
line[0], = ax.plot(dataset[:, 0], dataset[:, 1], 'k-', linewidth=0.8, label='Targeted', zorder=2) # 'b-' = blue line    
line[1], = ax.plot(X, Y, 'rD', markersize=6, fillstyle='none', markeredgecolor='r', markeredgewidth=0.8, label='Generated', zorder=3) # 'b-' = blue line    
plt.draw()
plt.show()
