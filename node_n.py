import numpy as np
from copy import deepcopy


class Node:

    # global id identifier
    __id = 0

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.inputs = 0
        self.left = None
        self.right = None
        self.id = Node.__id
        Node.__id += 1

    def update_fitness(self, error_function, X, y):
        self.fitness = error_function(self.output(X), y)

    def add_left_node(self, node):
        node.parent = self
        self.left = node

    def add_right_node(self, node):
        node.parent = self
        self.right = node

    def remove_left_node(self, node):
        node.parent = None
        self.left = None

    def remove_right_node(self, node):
        node.parent = None
        self.right = None

    def equation(self):
        equation = "("
        if self.left: equation += self.left.equation()
        equation += str(self)
        if self.right: equation += self.right.equation()
        equation += ")"
        return equation

    def sub_nodes(self):
        nodes = []
        nodes.append(self)
        if self.left: nodes = nodes + self.left.sub_nodes()
        if self.right: nodes = nodes + self.right.sub_nodes()
        return nodes

    def sub_tree(self):
        subtree = deepcopy(self)
        subtree.update_id()
        if self.left: self.left.update_id()
        if self.right: self.right.update_id()
        return subtree

    def update_id(self):
        self.id = Node.__id
        Node.__id += 1

    def depth(self):
        depth_left = 0
        depth_right = 0
        if self.left: depth_left = self.left.depth()
        if self.right: depth_right = self.right.depth()
        return max(depth_left, depth_right) + 1

    