import os
import sys
from pathlib import Path

functions = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
# functions = ['f3', 'f4', 'f5', 'f6', 'f7', 'f8']
# functions = ['f2']
population = 40
exp = '06'
iteration_num = 50

working_dir = Path(__file__).parent.parent.absolute()
print(working_dir)
dirpath = str(working_dir) + '\\tests\\data\\' + exp + '\\' + str(population) + '\\'
print(dirpath)

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

for i in range(len(functions)):
    fullpath = dirpath + functions[i] + '.dat'
    for j in range(iteration_num):
        print(f'Function: {functions[i]}\t | Test: {j}')
        command = f'dotnet run {functions[i]} {population} >> {fullpath}'
        os.system(command)
    print(" ")

import winsound
winsound.Beep(500, 250)
