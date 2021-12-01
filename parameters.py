class Par:
    POP_SIZE = 0
    MAX_EVAL = 0
    MAX_GEN = 0
    INIT_MIN_DEPTH = 0
    INIT_MAX_DEPTH = 0
    MAX_DEPTH = 0
    DOMAIN_X = []
    DOMAIN_Y = []
    VAR_NUM = 0
    POINT_NUM = 0
    OPERATORS = ['+', '-', '*', '/']
    FUNCTIONS = ['sin', 'cos', 'exp', 'rlog']
    VARIABLES = ['x0']
    CONSTANTS = range(1, 2)



    @classmethod
    def TARGET_FUNC(cls, *inputs):
        pass

