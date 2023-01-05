"""Node class.

The :mod:`Node` class represent the base for other functional and
terminal nodes used in forming programmes.
"""

import numpy as np
from copy import deepcopy


class Node:
    """Node class.

    Attributes:
        __id (int): Global id identifier.
        fitness (float): Current node tree's fitness value.
        parent (Node): Current node's parent (if found).
        arity (int): Arity or the number of arguments or operands.
        left (Node): Node's left child.
        right (Node): Node's right child.
        id (int): Current node's id identifier.
    """

    # global id identifier
    __id = 0

    def __init__(self):
        """initializing method."""
        self.fitness = np.inf
        self.parent = None
        self.arity = 0
        self.left = None
        self.right = None
        self.id = Node.__id
        Node.__id += 1

    def update_fitness(self, error_function, X, y):
        """Updating fitness method.

        Args:
            error_function (function): Function used for calculating fitness.
        """
        self.fitness = error_function(self.output(X), y)

    def add_left_node(self, node):
        """Adds node as a left child of the current node.

        Args:
            node (Node): Child node.
        """
        node.parent = self
        self.left = node

    def add_right_node(self, node):
        """Adds node as a right child of the current node.

        Args:
            node (Node): Child node.
        """
        node.parent = self
        self.right = node

    def remove_left_node(self, node):
        """Removes left child node.

        Args:
            node (Node): Child node.
        """
        node.parent = None
        self.left = None

    def remove_right_node(self, node):
        """Removes right child node.

        Args:
            node (Node): Child node.
        """
        node.parent = None
        self.right = None

    def equation(self):
        """Prints current node's tree as an equation.

        Returns:
            Current node's tree equation.
        """
        equation = "("
        if self.left:
            equation += self.left.equation()
        equation += str(self)
        if self.right:
            equation += self.right.equation()
        equation += ")"
        return equation

    def sub_nodes(self):
        """Finds all subnodes in current node's tree.

        Returns:
            List of subnode in current node's tree.
        """
        nodes = []
        nodes.append(self)
        if self.left:
            nodes = nodes + self.left.sub_nodes()
        if self.right:
            nodes = nodes + self.right.sub_nodes()
        return nodes

    def sub_tree(self):
        """Creates a subtree instance from current node's tree.

        Returns:
            Subtree instance.
        """
        subtree = deepcopy(self)
        subtree.update_id()
        if self.left:
            self.left.update_id()
        if self.right:
            self.right.update_id()
        return subtree

    def update_id(self):
        """Updates current node's id"""
        self.id = Node.__id
        Node.__id += 1

    def depth(self):
        """Caclulates the depth of current node's tree.

        Returns:
            Depth value of current node's tree.
        """
        depth_left = 0
        depth_right = 0
        if self.left:
            depth_left = self.left.depth()
        if self.right:
            depth_right = self.right.depth()
        return max(depth_left, depth_right) + 1
