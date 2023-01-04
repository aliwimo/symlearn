"""Global parameters.

The :mod:`parameters` module contains a set of different global parameters that is 
used while solving problems.
"""

class Parameters:
    """Parameters class.

    Attributes:
        CONSTANTS (list): List of constants used in nodes
        FEATURES (int): Number of unique variables
        CONSTANTS_TYPE (string): Constant type modifier
        EXPORT_EXT (string): Export graph extension
    """
    CONSTANTS = [0, 1]
    FEATURES = 1
    CONSTANTS_TYPE = 'range'
    EXPORT_EXT = 'pdf'

