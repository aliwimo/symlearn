import os
import sys
from pathlib import Path

# population = 50
iteration_num = 30
# exp = 'Box'
# function = 'Box'

# working_dir = Path(__file__).parent.parent.absolute()
# dirpath = str(working_dir) + '\\tests\\data\\' + exp + '\\' + str(population) + '\\'

# if not os.path.exists(dirpath):
#     os.makedirs(dirpath)

for i in range(iteration_num):
    # fullpath = dirpath + 'test_' + str(i) + '.dat'
    command = f'echo {i + 1}'
    os.system(command)
    command = f'python main.py'
    os.system(command)
    command = 'echo --------------------------------'
    os.system(command)
