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

    def get_id(self):
        return self.id

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

    def get_id(self):
        return self.id

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

    def get_id(self):
        return self.id

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
        return '/'

    def get_id(self):
        return self.id

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        sign_X1 = np.sign(X1)   # numpy.sign: X>=1 returns 1, X==0 returns 0, X<=1 returns -1 
        sign_X1[sign_X1 == 0] = 1
        return X0 / sign_X1
        # sign_X1 = np.sign(X1)
        # sign_X1[sign_X1 == 0] = 1
        # return np.multiply(sign_X1, X0) / ( 1e-6 + np.abs(X1))


class Sin(Node):
    def __init__(self):
        super(Sin, self).__init__()
        self.inputs = 1
        self.type = 'function'

    def __repr__(self):
        return 'sin'

    def get_id(self):
        return self.id

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

    def get_id(self):
        return self.id

    def output(self, X):
        X0 = self.right.output(X)
        return np.cos(X0)


class Pow(Node):
    def __init__(self):
        super(Pow, self).__init__()
        self.inputs = 2

    def __repr__(self):
        return 'pow'

    def get_id(self):
        return self.id

    def output(self, X):
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return np.power(X0, X1)


class Exp(Node):
    def __init__(self):
        super(Exp, self).__init__()
        self.inputs = 1

    def __repr__(self):
        return 'exp'

    def get_id(self):
        return self.id

    def output(self, X):
        X0 = self.right.output(X)
        return np.exp(X0)


class Rlog(Node):
    def __init__(self):
        super(Rlog, self).__init__()
        self.inputs = 1

    def __repr__(self):
        return 'rlog'

    def get_id(self):
        return self.id

    def output(self, X):
        X0 = self.right.output(X)
        return np.log(np.abs(X0) + 1e-6)

class Variable(Node):
    def __init__(self):
        super(Variable, self).__init__()
        self.rank = np.random.randint(low=0, high=Parameters.FEATURES)
        self.type = 'feature'

    def __repr__(self):
        return 'x' + str(self.rank)

    def get_id(self):
        return self.id

    def output(self, X):
        return X[:, self.rank]


class Constant(Node):
    def __init__(self):
        super(Constant, self).__init__()
        self.value = np.random.randint(low = 1, high = 2)
        self.type = 'terminal'

    def __repr__(self):
        return str(self.value)

    def get_id(self):
        return self.id

    def output(self, X):
        return np.array([self.value] * X.shape[0])
