import numpy as np
from tree import Tree
from parameters import Par
from firefly import Firefly
from sys import argv
from random import random


np.seterr(all='ignore')

X = np.random.uniform(-1, 1, 20).reshape(20, 1)
Y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0])

# X = np.random.uniform(0, 1, 20).reshape(20, 1)
# Y = (np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2))

# X = np.random.uniform(0, 1, (20, 2)).reshape(20, 2)
# Y = (np.sin(X[:, 0]) + np.sin(X[:, 1] ** 2))

f1 = {'X': [-1, 1],     'Y': [0, 1],    'P': 20,   'V': 1}  # f1: 20p [-1, 1]        GP, ABCP: [0.30, 0.11]
f2 = {'X': [-1, 1],     'Y': [0, 1],    'P': 20,   'V': 1}  # f2: 20p [-1, 1]        GP, ABCP: [0.40, 0.19]
f3 = {'X': [0, 1],      'Y': [0, 1],    'P': 20,   'V': 1}  # f3: 20p [0, 1]         GP, ABCP: [0.11, 0.21]
f4 = {'X': [0, np.pi/2],'Y': [0, 1],    'P': 20,   'V': 1}  # f4: 20p [0, pi/2]      GP, ABCP: [0.27, 0.55]
f5 = {'X': [0, 2],      'Y': [0, 1],    'P': 20,   'V': 1}  # f5: 20p [0, 2]         GP, ABCP: [0.15, 0.24]
f6 = {'X': [0, 4],      'Y': [0, 1],    'P': 20,   'V': 1}  # f6: 20p [0, 4]         GP, ABCP: [0.25, 0.17]
f7 = {'X': [0, 1.1],    'Y': [0, 1],    'P': 20,  'V': 2}  # f7: 100p [0, 1]*[0, 1] GP, ABCP: [1.67, 0.62]
f8 = {'X': [0, 1.1],    'Y': [0, 1],    'P': 20,  'V': 2}  # f8: 100p [0, 1]*[0, 1] GP, ABCP: [1.19, 0.69]

selected            = f1
Par.POP_SIZE        = 25
Par.MAX_EVAL        = 25000
Par.MAX_GEN         = 1000
Par.INIT_MIN_DEPTH  = 0
Par.INIT_MAX_DEPTH  = 6
Par.MAX_DEPTH       = 15

def target_function(x, y=None):
    if   selected == f1: return x*x*x*x + x*x*x + x*x + x
    elif selected == f2: return x**5 + x**4 + x**3 + x**2 + x
    elif selected == f3: return np.sin(x) + np.sin(x + (x**2))
    elif selected == f4: return np.sin(x**2) * np.cos(x) - 1 
    elif selected == f5: return np.log(x + 1) + np.log((x**2) + 1)
    elif selected == f6: return np.sqrt(x)
    elif selected == f7: return np.sin(x) + np.sin(y ** 2)
    elif selected == f8: return 2 * np.sin(x) * np.cos(y)

Par.DOMAIN_X        = selected['X']
Par.DOMAIN_Y        = selected['Y']
Par.POINT_NUM       = selected['P']
# Par.VAR_NUM         = selected['V']
Par.VAR_NUM         = 1
Par.TARGET_FUNC     = target_function
Par.X               = X
Par.Y               = Y
Tree.OPER_FUN_RATE  = 0.5
Tree.TERMINAL_RATE  = 0.5

# create variables list
VARIABLES = []
for i in range(Par.VAR_NUM):
    var = 'x' + str(i)
    VARIABLES.append(var)
Par.VARIABLES = VARIABLES

if len(argv) > 1:
    Par.POP_SIZE = int(argv[1])

ffp = Firefly(alpha=1, beta=2, gamma=3)
ffp.run()

# x = Tree()
# x.create_tree('full', Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)

# x.draw_tree('test', 'label')