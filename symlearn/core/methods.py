"""Methods used in evaluation process.

The :mod:`methods` module contains a set of different methods that is 
used for modifying programmes during optimization process.
"""

import numpy as np
from random import random, choice
from graphviz import Digraph, Source  # install version 0.16
from symlearn.core.parameters import Parameters
from symlearn.core.node import Node


class Methods:
    """Methods used in evaluation process.

    This class contains a set of different methods that is 
    used for modifying programmes during optimization process.
    """

    @classmethod
    def generate_population(cls,
                            pop_size,
                            initial_min_depth,
                            initial_max_depth,
                            expressions,
                            terminals):
        """Generate the initial population for population-based models.

        This method generates population using `ramped-half-and-half` 
        method which is a combination of `full` and `grow` forming
        methods. 

        Args:
            pop_size (int): The size of the population to generate.
            initial_min_depth (int): The minimum depth of the trees to generate.
            initial_max_depth (int): The maximum depth of the trees to generate.
            expressions (list): Set of different expression used in tree
            terminals (list): Set of different terminals used in tree

        Returns:
            A list of trees that represent the population
        """
        population = []
        # generate with full method
        for _ in range(pop_size // 2):
            individual = cls.generate_individual(
                'full', initial_min_depth, initial_max_depth, expressions, terminals)
            population.append(individual)
        # generate with grow method
        for _ in range((pop_size // 2), pop_size):
            individual = cls.generate_individual(
                'grow', initial_min_depth, initial_max_depth, expressions, terminals)
            population.append(individual)
        return population

    @classmethod
    def generate_individual(cls,
                            method,
                            initial_min_depth,
                            initial_max_depth,
                            expressions,
                            terminals,
                            current_depth=0) -> Node:
        """Generate an individual tree (node).

        This method generates an individual using either 
        `full` or`grow` forming method.

        Args:
            method (string): The size of the generated population
            initial_min_depth (int): Minimum trees forming height
            initial_max_depth (int): Maximum trees forming height
            expressions (list): Set of different expression used in tree
            terminals (list): Set of different terminals used in tree
            current_depth (int): Tree's depth controlling argument

        Returns:
            Individual tree's head (node).
        """
        if method == 'full':
            if current_depth < initial_max_depth - 1:
                node = choice(expressions)()
            else:
                node = choice(terminals)()
        elif method == 'grow':
            if current_depth < initial_min_depth:
                node = choice(expressions)()
            elif initial_min_depth <= current_depth < (initial_max_depth - 1):
                if random() > 0.5:
                    node = choice(expressions)()
                else:
                    node = choice(terminals)()
            else:
                node = choice(terminals)()
        # create left and right branches
        if node.arity == 2:
            child_1 = cls.generate_individual(
                method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            child_2 = cls.generate_individual(
                method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            node.add_left_node(child_1)
            node.add_right_node(child_2)
        elif node.arity == 1:
            child = cls.generate_individual(
                method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            node.add_right_node(child)
        return node

    @classmethod
    def rank_trees(cls, trees, errors, reverse=False):
        """Ranks the trees based on the errors.

        Args:
            trees (List[Node]): list of trees to be ranked
            errors (List[float]): list of corresponding errors for the given trees
            reverse (bool, optional): Flag indicating whether the ranking should be in descending order or not. Default is False.

        Returns:
            List of ranked trees, and corresponding errors.
        """
        combined = list(zip(errors, trees))
        combined.sort(key=lambda x: x[0], reverse=reverse)
        errors, trees = zip(*combined)
        return list(trees), list(errors) 

    @classmethod
    def export_graph(cls, root: Node, file_name, label):
        """Exports the graph of a tree.

        This method exports the graph of a tree.

        Args:
            root (node): Exported tree's head (node)
            filename (string): Exported file name
            label (string): Label displayed in the exported graph
        """
        graph = [Digraph()]
        graph[0].attr(kw='graph', label=label)
        cls.draw_node(graph, root)
        Source(graph[0], filename=file_name + '.gv',
               format=Parameters.EXPORT_EXT).render()

    @classmethod
    def draw_node(cls, graph, root: Node):
        """This is an `export_graph` sub function.

        This method draws single tree's node.

        Args:
            graph (node): The graph that contains the node
            root (node): Exported tree's head (node)
        """
        graph[0].node(str(root.id), str(root))
        if root.left:
            graph[0].edge(str(root.id), str(root.left.id))
            cls.draw_node(graph, root.left)
        if root.right:
            graph[0].edge(str(root.id), str(root.right.id))
            cls.draw_node(graph, root.right)
