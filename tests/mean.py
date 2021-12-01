#         F1      F2      F3      F4      F5      F6      F7      F8
# GP      0.30    0.40    0.11    0.27    0.15    0.25    1.67    1.19  
# ABCP    0.11    0.19    0.21    0.55    0.24    0.17    0.62    0.69

# exp 01:   Firefly.alpha			= Na;
#           Firefly.beta			= Na;
#           Firefly.gamma			= Na;
# note: results of standard FP without subs process
# --------------------------------------------
# exp 02:   Firefly.alpha			= 0.1;
#           Firefly.beta			= 0.75;
#           Firefly.gamma			= 1.5;
# note: second method with sub process
# --------------------------------------------
# exp 03:   Firefly.alpha			= Na;
#           Firefly.beta			= Na;
#           Firefly.gamma			= Na;
# note: removed the condition of inner foor loop
# --------------------------------------------
# exp 04:   Firefly.alpha			= Na;
#           Firefly.beta			= Na;
#           Firefly.gamma			= Na;
# note: loops only for n in the inner loop (j loop)
# --------------------------------------------
# exp 04:   Firefly.alpha			= Na;
#           Firefly.beta			= Na;
#           Firefly.gamma			= Na;
# note: loops only for n in the inner loop (j loop)



import os
import sys
from pathlib import Path
import numpy as np

experiment = '04'

# working_dir = os.path.dirname(sys.argv[0]) + '/data/' + experiment
working_dir = str(Path(__file__).parent.absolute()) + '\\data\\' + experiment
print(working_dir)
sub_dirs = os.listdir(working_dir)

for i in range(8):
    print(f'\t f{i+1}', end='')
print('')

for i in range(len(sub_dirs)):
    sub_dirs[i] = str("{0:0=4d}".format(int(sub_dirs[i])))
sub_dirs.sort()

for x in sub_dirs:
    dat_files = os.listdir(working_dir)
    print(f'{int(x)}:', end='\t')
    for i in range(8):
        try:
            values = np.loadtxt(working_dir + '\\' + str(int(x)) + '\\f' + str(i + 1) + '.dat')
            print(f'{round(np.mean(values), 2)}', end='\t')
        except:
            print(f'---', end='\t')
    print('')
