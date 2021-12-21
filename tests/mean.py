from pathlib import Path
import numpy as np

population = 100
data_path = str(Path(__file__).parent.absolute()) + '\\data\\' + str(population) + '\\box.dat'
print(data_path)
data = np.loadtxt(data_path)

mean = np.mean(data)
std = np.std(data)
best = np.min(data)

print(f'Best: {best}')
print(f'Mean: {mean}')
print('Std: {:f}'.format(std))

