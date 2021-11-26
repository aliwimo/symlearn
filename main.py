from parameters import Parameters
from expressions import *
from functions import Functions

X = np.array([[1, 2, 3], [4, 5, 6]])

Parameters.POP_SIZE         = 4
Parameters.INIT_MIN_DEPTH   = 1
Parameters.INIT_MAX_DEPTH   = 3
Parameters.MAX_DEPTH        = 15
Parameters.EXPRESSIONS      = [Add, Sub]
Parameters.TERMINALS        = [Constant, Variable]


s = Functions.generate_individual('grow')
t = Functions.generate_individual('grow')

Functions.export_graph(s, 'Source', 'Source')
print(s.equation())
Functions.export_graph(t, 'Target_before', 'Target_before')
print(t.equation())

n = Functions.share(s, t)
Functions.export_graph(n, 'Target_after', 'Target_after')
print(n.equation())
