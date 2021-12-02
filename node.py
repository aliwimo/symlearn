from random import choice
from parameters import Parameters

class Node:
    __id = 0

    def __init__(self, value=None, node_type=None, parent_id=None):
        self.value = value
        self.node_type = node_type
        self.parent_id = parent_id
        self.rank = 0
        self.id = Node.__id
        Node.__id += 1

    def set_value(self, node_type):
        if node_type == "function":
            self.value = choice(Parameters.FUNCTIONS)
        elif node_type == "operator":
            self.value = choice(Parameters.OPERATORS)
        elif node_type == "variable":
            self.value = choice(Parameters.VARIABLES)
            self.rank = int(''.join(filter(str.isdigit, self.value)))
        else:
            self.value = choice(Parameters.CONSTANTS)        
        self.node_type = node_type

    def draw(self, graph):
        if not self.parent_id is None:
            graph[0].edge(str(self.parent_id), str(self.id))
        graph[0].node(str(self.id), str(self.value))