from parameters import Parameters
from expressions import *
from functions import Functions

X = np.array([[1, 2, 3], [4, 5, 6]])

Parameters.POP_SIZE         = 4
Parameters.INIT_MIN_DEPTH   = 2
Parameters.INIT_MAX_DEPTH   = 4
Parameters.MAX_DEPTH        = 15
Parameters.EXPRESSIONS      = [Add, Sub]
Parameters.TERMINALS        = [Constant, Variable]


n1 = Add()
n2 = Add()
c1 = Constant()
c2 = Constant()
n3 = Add()
c3 = Constant()
v1 = Variable()
v2 = Variable()

n1.add_child(n2)
n1.add_child(n3)
n2.add_child(c1)
n2.add_child(c2)
n3.add_child(v1)
n3.add_child(c3)

print(n1.equation())
Functions.export_graph(n1, 'N_Before', 'Before')

n1.remove_child(n3)
n1.add_child(v2)
print(n1.equation())
Functions.export_graph(n1, 'N_After', 'After')
