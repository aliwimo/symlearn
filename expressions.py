import numpy as np
from node import Node

class Add(Node):
    def __init__(self):
        super(Add, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '+'

    def output(self, X):
        X0 = self.children[0].output(X)
        X1 = self.children[1].output(X)
        return X0 + X1

class Sub(Node):
    def __init__(self):
        super(Sub, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '-'

    def output(self, X):
        X0 = self.children[0].output(X)
        X1 = self.children[1].output(X)
        return X0 - X1


class Variable(Node):
    def __init__(self):
        super(Variable, self).__init__()
        self.rank = np.random.randint(low = 0, high = 3)
        self.type = 'feature'

    def __repr__(self):
        return 'x' + str(self.rank)

    def output(self, X):
        return X[:, self.rank]


class Constant(Node):
    def __init__(self):
        super(Constant, self).__init__()
        self.value = np.random.randint(low = 0, high = 3)
        self.type = 'terminal'

    def __repr__(self):
        return str(self.value)

    def output(self, X):
        return self.value
