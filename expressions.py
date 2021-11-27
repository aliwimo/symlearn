import numpy as np
from node import Node
from parameters import Parameters

class Add(Node):
    def __init__(self):
        super(Add, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '+'

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 + X1

class Sub(Node):
    def __init__(self):
        super(Sub, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '-'

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 - X1

class Mul(Node):
    def __init__(self):
        super(Mul, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '*'

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 * X1


class Div(Node):
    def __init__(self):
        super(Div, self).__init__()
        self.inputs = 2
        self.type = 'function'

    def __repr__(self):
        return '*'

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        sign_X1 = np.sign(X1)
        sign_X1[sign_X1 == 0] = 1
        return np.multiply(sign_X1, X0) / ( 1e-6 + np.abs(X1))


class Sin(Node):
    def __init__(self):
        super(Sin, self).__init__()
        self.inputs = 1
        self.type = 'function'

    def __repr__(self):
        return 'sin'

    def output(self, X):
        X0 = self.right.output(X)
        return np.sin(X0)


class Cos(Node):
    def __init__(self):
        super(Cos, self).__init__()
        self.inputs = 1
        self.type = 'function'

    def __repr__(self):
        return 'cos'

    def output(self, X):
        X0 = self.right.output(X)
        return np.cos(X0)



class Variable(Node):
    def __init__(self):
        super(Variable, self).__init__()
        self.rank = np.random.randint(low=0, high=Parameters.FEATURES)
        self.type = 'feature'

    def __repr__(self):
        return 'x' + str(self.rank)

    def output(self, X):
        return X[:, self.rank]


class Constant(Node):
    def __init__(self):
        super(Constant, self).__init__()
        self.value = np.random.randint(low = 1, high = 2)
        self.type = 'terminal'

    def __repr__(self):
        return str(self.value)

    def output(self, X):
        return np.array([self.value] * X.shape[0])
