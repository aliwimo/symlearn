import numpy as np

class Node:
    __id = 0

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.inputs = 0
        self.children = []
        self.id = Node.__id
        Node.__id += 1
    
    def output(self, X):
        return None

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def remove_child(self, node):
        if node in self.children:
            index = self.children.index(node)
            self.children.pop(index)

    def equation(self):
        eq = ''
        if self.inputs == 2:
            eq += '(' + self.children[0].equation() + ' '
            eq += self.__repr__();
            eq += ' ' + self.children[1].equation() + ')'
        elif self.inputs == 1:
            eq += self.__repr__();
            eq += '(' + self.children[0].equation() + ')'
        else:
            eq += self.__repr__()
        return str(eq)

    def draw_node(self, graph):
        graph[0].node(str(self.id), str(self))
        for n in self.children:
            graph[0].edge(str(self.id), str(n.id))
            n.draw_node(graph)

    
