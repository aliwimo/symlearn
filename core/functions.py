import numpy as np
from random import uniform
from core.node import Node
from core.parameters import Parameters

class Add(Node):
    """Addtion function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Add, self).__init__()
        self.inputs = 2
        self.type = 'expression'

    def __repr__(self):
        """representing method."""
        return '+'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 + X1

class Sub(Node):
    """Subtraction function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Sub, self).__init__()
        self.inputs = 2
        self.type = 'expression'

    def __repr__(self):
        """representing method."""
        return '-'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 - X1

class Mul(Node):
    """Multiplication function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Mul, self).__init__()
        self.inputs = 2
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return '*'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return X0 * X1

class Div(Node):
    """Division function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Div, self).__init__()
        self.inputs = 2
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return '/'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        sign_X1 = np.sign(X1)   # numpy.sign: X>=1 returns 1, X==0 returns 0, X<=1 returns -1 
        sign_X1[sign_X1 == 0] = 1
        return X0 / sign_X1

class Sin(Node):
    """Sine function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Sin, self).__init__()
        self.inputs = 1
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return 'sin'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.right.output(X)
        return np.sin(X0)

class Cos(Node):
    """Cosine function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Cos, self).__init__()
        self.inputs = 1
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return 'cos'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.right.output(X)
        return np.cos(X0)

class Rlog(Node):
    """Protected logarithm function.

    This function represents the protected natural logarithm 
    function that returns a very small number of inputs equals
    to zero.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Rlog, self).__init__()
        self.inputs = 1
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return 'rlog'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.right.output(X)
        sign_X0 = np.sign(X0)   # numpy.sign: X>=1 returns 1, X==0 returns 0, X<=1 returns -1 
        X0[sign_X0 == 0] = 1e-6
        # return np.log(np.abs(X0) + 1e-6)
        return np.log(np.abs(X0))

class Pow(Node):
    """Power function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Pow, self).__init__()
        self.inputs = 2
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return 'pow'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.left.output(X)
        X1 = self.right.output(X)
        return np.power(X0, np.abs(X1))
        # return np.power(X0, X1)

class Exp(Node):
    """Exponential function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
    """
    def __init__(self):
        """initializing method."""
        super(Exp, self).__init__()
        self.inputs = 1
        self.type = 'expression'
    
    def __repr__(self):
        """representing method."""
        return 'exp'

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        X0 = self.right.output(X)
        return np.exp(X0)

class Variable(Node):
    """Variable function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
        rank (int): Variable rank indicator
    """
    def __init__(self):
        """initializing method."""
        super(Variable, self).__init__()
        self.inputs = 0
        self.type = 'variable'
        self.rank = np.random.randint(low=0, high=Parameters.FEATURES)

    def __repr__(self):
        """representing method."""
        return 'x' + str(self.rank)

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        if X.shape[0] == 1:
            return np.array([1] * X.shape[0])
        else:
            return X[:, self.rank]

class Constant(Node):
    """Constant function.

    Attributes:
        arity (int): The number of arguments or operands
        type (string): Type indicator of the function
        value (int): Value of the constant
    """
    def __init__(self):
        """initializing method."""
        super(Constant, self).__init__()
        self.inputs = 0
        self.type = 'terminal'
        if Parameters.CONSTANTS_TYPE == 'integer':
            self.value = np.random.randint(low=Parameters.CONSTANTS[0], high=Parameters.CONSTANTS[1])
        elif Parameters.CONSTANTS_TYPE == 'range':
            self.value = uniform(Parameters.CONSTANTS[0], Parameters.CONSTANTS[1]) 

    def __repr__(self):
        """representing method."""
        return str(round(self.value, 2))

    def output(self, X):
        """Output method.
        
        Args:
            X (list | float | int): input variable

        Returns:
            The value of the function after applying it to the inputs
        """
        return np.array([self.value] * X.shape[0])
        