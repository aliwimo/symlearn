import numpy as np
import matplotlib.pyplot as plt
y1 = [51, 33, 19, 0, 0, 22, 9, 37]        # 0%
# y1 = [55, 44, 37, 9, 18, 48, 12, 40]      # <20%
y2 = [74, 46, 33, 1, 1, 12, 28, 13]       # 0%
# y2 = [78, 56, 77, 6, 82, 32, 44, 55]      # <20%
X = np.arange(1, 9)
functions = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']

# preparing plot
ax = plt.axes()

# showing grid 
ax.grid(linestyle=':', linewidth=1, alpha=1, zorder=1)

# set the Label of X and Y axis
plt.ylabel("Number of Solutions")
plt.ylim([0, 100])

gap = 0.10
width = 0.10

# ax.bar(X - gap, y1, color='black', width=width, zorder=10, label='FP')
# ax.bar(X - gap, y1, hatch='////', edgecolor='black', color='blak', width=width, zorder=10, label='FP')
ax.bar(X - gap, y1, color='gray', width=width, zorder=10, label='FP')
ax.bar(X + gap, y2, color='black', width=width, zorder=12, label='DFP')

for i, x in enumerate(X):
    # ax.text(X[i] - (gap/2 + width/3), y1[i] + 1, str(y1[i]), horizontalalignment='center', color='black')
    ax.text(X[i] - (gap/2 + width/2), y1[i] + 2, str(y1[i]), horizontalalignment='center', color='black')
    ax.text(X[i] + (gap/2 + width/2), y2[i] + 2, str(y2[i]), horizontalalignment='center', color='black')

# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)

ax.set_xticklabels(functions)

# show the legend of the graphs 
ax.legend()

# show graphes
plt.draw()
plt.show()