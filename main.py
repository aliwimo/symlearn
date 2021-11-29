from parameters import Parameters
from expressions import *
from functions import Functions
from firefly import Firefly

# X = np.array([[1, 2, 3], [4, 5, 6]])


X = np.random.uniform(-1, 1, 20).reshape(20, 1)
Y = (X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0])

Parameters.POP_SIZE         = 50
Parameters.MAX_EVAL         = 25000
Parameters.INIT_MIN_DEPTH   = 0
Parameters.INIT_MAX_DEPTH   = 3
Parameters.MAX_DEPTH        = 15
# Parameters.EXPRESSIONS      = [Add, Sub, Mul, Div, Sin, Cos, Exp, Rlog]
# Parameters.EXPRESSIONS      = [Add, Sub, Mul, Div]
Parameters.EXPRESSIONS      = [Add, Sub, Mul, Div, Sin, Cos, Rlog]
Parameters.TERMINALS        = [Constant, Variable]
Parameters.FEATURES         = X.shape[1]

ffp = Firefly(X, Y)
ffp.run()


# node1 = Functions.generate_individual('full')
# print(node1.equation())
# Functions.export_graph(node1, 'node1', 'node1')
# node2 = Functions.generate_individual('full')
# print(node2.equation())
# Functions.export_graph(node2, 'node2', 'node2')

# node3 = Functions.share(node1, node2)
# print(node3.equation())
# Functions.export_graph(node3, 'node3', 'node3')