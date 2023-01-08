"""Node class.

The :mod:`Node` class represent the base for other functional and
terminal nodes used in forming programmes.
"""

import numpy as np
from copy import deepcopy


class Node:
    """
    A base class representing a node in a tree structure.

    This class has a class attribute `__id` that is used to generate a unique
    integer id for each node when it is initialized. The `__init__` method sets
    the node's id, arity, fitness, and parent attributes, and initializes the
    left and right children to None. The `update_fitness` method updates the
    node's fitness attribute using the error function passed as an argument.

    Attributes:
        __id (int): Global id identifier.
        fitness (float): Current node tree's fitness value.
        parent (Node): A reference to the parent node of the node (if found).
        arity (int): The number of children that the node has.
        left (Node): Node's left child.
        right (Node): Node's right child.
        id (int): Current node's id identifier.
    """

    # global id identifier
    __id = 0

    def __init__(self):
        """Initializes the Node.

        Sets the id, arity, fitness, and parent attributes, and initializes the
        left and right children to None.
        """
        self.fitness = np.inf
        self.parent = None
        self.arity = 0
        self.left = None
        self.right = None
        self.id = Node.__id
        Node.__id += 1

    def update_fitness(self, error_function, X, y):
        """Updates the fitness of the node.

        Args:
            error_function: A function that takes in the output of the node's
                mathematical expression tree and the target output and returns a
                scalar error value.
            X: The input to the mathematical expression tree.
            y: The target output.
        """
        self.fitness = error_function(self.output(X), y)

    def add_left_node(self, node):
        """Adds node as the left child of the current node.

        Args:
            node: The node to be added as the left child.
        """
        node.parent = self
        self.left = node

    def add_right_node(self, node):
        """Adds node as the right child of the current node.

        Args:
            node: The node to be added as the right child.
        """
        node.parent = self
        self.right = node

    def remove_left_node(self, node):
        """Removes the left child of the current node.

        Args:
            node: The node to be removed as the left child.
        """
        node.parent = None
        self.left = None

    def remove_right_node(self, node):
        """Removes the right child of the current node.

        Args:
            node: The node to be removed as the right child.
        """
        node.parent = None
        self.right = None

    def equation(self):
        """Represents current node's tree as an equation.

        Returns:
            A string representation of the mathematical expression represented by the tree rooted at the current node.
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
        """Finds all subnodes in the tree rooted at the current node.

        Returns:
            A list of all subnodes in the tree rooted at the current node.
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
            A deep copy of the tree rooted at the current node.
        """
        subtree = deepcopy(self)
        subtree.update_id()
        if self.left:
            self.left.update_id()
        if self.right:
            self.right.update_id()
        return subtree

    def update_id(self):
        """Updates the id attribute of the current node to a new unique integer."""
        self.id = Node.__id
        Node.__id += 1

    def depth(self):
        """Caclulates the depth of current node's tree.

        Returns:
            An integer representing the depth of the tree rooted at the current node.
        """
        depth_left = 0
        depth_right = 0
        if self.left:
            depth_left = self.left.depth()
        if self.right:
            depth_right = self.right.depth()
        return max(depth_left, depth_right) + 1
