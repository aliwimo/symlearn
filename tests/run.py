import os
import sys
from pathlib import Path

population = 50
iteration_num = 20
exp = '01'
function = 'f3'

working_dir = Path(__file__).parent.parent.absolute()
print(working_dir)
dirpath = str(working_dir) + '\\tests\\data\\' + exp + '\\' + str(population) + '\\'
print(dirpath)

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

fullpath = dirpath + function + '.dat'
for i in range(iteration_num):
    print(f'Test {i + 1}')
    command = f'python main.py >> {fullpath}'
    os.system(command)
print(" ")
