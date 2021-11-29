import numpy as np
from copy import deepcopy

class Node:
    __id = 0

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.inputs = 0
        self.left = None
        self.right = None
        self.position = None
        # self.id = self.update_id()
        self.id = Node.__id
        Node.__id += 1
    
    def output(self, X):
        return None

    def add_left_child(self, node):
        self.left = node
        node.position = 'left'
        node.parent = self

    def add_right_child(self, node):
        self.right = node
        node.position = 'right'
        node.parent = self

    def equation(self):
        eq = ''
        if self.inputs == 2:
            eq += '(' + self.left.equation() + ' '
            eq += self.__repr__();
            eq += ' ' + self.right.equation() + ')'
        elif self.inputs == 1:
            eq += self.__repr__();
            eq += '(' + self.right.equation() + ')'
        else:
            eq += self.__repr__()
        return str(eq)

    def depth(self):
        depth_left = 0
        depth_right = 0
        if self.left: depth_left = self.left.depth()
        if self.right: depth_right = self.right.depth()
        return max(depth_left, depth_right) + 1

    def copy_subtree(self):
        subtree = deepcopy(self)
        subtree.update_id()
        if subtree.left: subtree.left.update_id()
        if subtree.right: subtree.right.update_id()
        return subtree

    def update_id(self):
        self.id = Node.__id
        Node.__id += 1

    def draw_node(self, graph):
        graph[0].node(str(self.id), str(self))
        if self.left:
            graph[0].edge(str(self.id), str(self.left.id))
            self.left.draw_node(graph)
        if self.right:
            graph[0].edge(str(self.id), str(self.right.id))
            self.right.draw_node(graph)



    
