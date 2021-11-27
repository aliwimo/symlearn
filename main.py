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
Parameters.INIT_MAX_DEPTH   = 6
Parameters.MAX_DEPTH        = 15
Parameters.EXPRESSIONS      = [Add, Sub, Mul, Div, Sin, Cos]
Parameters.TERMINALS        = [Constant, Variable]
Parameters.FEATURES         = X.shape[1]

ffp = Firefly(X, Y)
ffp.run()



# s = Functions.generate_individual('full')
# Functions.export_graph(s, 'Source', 'Source')
# output = s.output(X)
# print(output)

# error = np.sum(np.abs(output - Y))
# print(error)


# t = Functions.generate_individual('grow')

# Functions.export_graph(s, 'Source', 'Source')
# print(s.equation())
# Functions.export_graph(t, 'Target_before', 'Target_before')
# print(t.equation())

# n = Functions.share(s, t)
# Functions.export_graph(n, 'Target_after', 'Target_after')
# print(n.equation())
