"""Methods used in evaluation process.

The :mod:`methods` module contains a set of different methods that is 
used for modifying programmes during optimization process.
"""

from random import random, choice
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph, Source  # install version 0.16
from symlearn.core.parameters import Parameters
from symlearn.core.node import Node
from symlearn.core.functions import Constant


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
    def share(cls, source: Node, target: Node):
        """Performs sharing operation between two trees.

        This method takes an instance subtree from source and glues
        it to the target.

        Args:
            source (node): The source tree head (node)
            target (node): The target tree head (node)

        Returns:
            The new generated tree's head (node).
        """
        source_nodes = source.sub_nodes()

        if len(source_nodes) > 1:
            if random() < 0.9:
                is_function = False
                while not is_function:
                    instance_node = choice(source_nodes).sub_tree()
                    if instance_node.arity >= 1:
                        is_function = True
            else:
                instance_node = choice(source_nodes).sub_tree()
        else:
            instance_node = choice(source_nodes).sub_tree()
        target_nodes = target.sub_nodes()
        removed_node = choice(target_nodes)
        parent = removed_node.parent
        if parent:
            if removed_node.parent.left == removed_node:
                parent.remove_left_node(removed_node)
                parent.add_left_node(instance_node)
            elif removed_node.parent.right == removed_node:
                parent.remove_right_node(removed_node)
                parent.add_right_node(instance_node)
            return target
        else:
            return instance_node

    @classmethod
    def change_node(cls, source: Node, nodes_pool):
        """Performs replacement operation in one tree.

        This method takes selects a random node in the source
        tree and replaces it with a same-arity node.

        Args:
            source (node): The source tree head (node)
            nodes_pool (list): List of possible replacement nodes

        Returns:
            The new generated tree's head (node).
        """
        source_nodes = source.sub_nodes()
        selected_node = choice(source_nodes)
        same_arity = False
        while not same_arity:
            new_node = choice(nodes_pool)()
            if new_node.arity == selected_node.arity:
                same_arity = True
        parent = selected_node.parent
        if parent:
            if selected_node.parent.left == selected_node:
                parent.remove_left_node(selected_node)
                parent.add_left_node(new_node)
            elif selected_node.parent.right == selected_node:
                parent.remove_right_node(selected_node)
                parent.add_right_node(new_node)

        if selected_node.arity == 2:
            new_node.add_left_node(deepcopy(selected_node.left))
            new_node.add_right_node(deepcopy(selected_node.right))
        elif selected_node.arity == 1:
            new_node.add_right_node(deepcopy(selected_node.right))
        return source

    @classmethod
    def rank_trees(cls, trees, fitnesses, is_reversed=False):
        """Ranks trees according to fitness rank.

        This method ranks trees after sorting the fitness values.

        Args:
            trees (list): List of trees' heads (nodes)
            fitnesses (list): List of fitness values of trees
            is_reversed (bool): Reversed sorting modifier

        Returns:
            Trees: List of sorted trees' heads (nodes)
            fitnesses: List of sorted fitness values
        """
        sorted_indices = np.argsort(fitnesses)
        if not is_reversed:
            sorted_indices = np.flip(sorted_indices)
        fitnesses.sort(reverse=not is_reversed)
        temp_trees = trees.copy()
        for (m, n) in zip(range(len(trees)), sorted_indices):
            trees[m] = temp_trees[n]
        return trees, fitnesses

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

    @classmethod
    def simplify(cls, root: Node):
        """Simplifies tree by combining nodes.

        This method simplifies trees by combining tree's branches
        that does not haveany variable nodes.

        Args:
            root (node): Simplified tree's head (node)

        Returns:
            A boolean modifier that checks if current node has a
            variable child.
        """
        has_variables = False
        if root.type == 'variable':
            has_variables = True
        if root.left:
            has_variables = cls.simplify(root.left)
        if root.right:
            has_variables = cls.simplify(root.right)
        if not has_variables:
            result = root.output(np.array([[1]]))
            new_node = Constant()
            new_node.value = result[0]

            parent = root.parent
            if parent:
                if root.parent.left == root:
                    parent.remove_left_node(root)
                    parent.add_left_node(new_node)
                elif root.parent.right == root:
                    parent.remove_right_node(root)
                    parent.add_right_node(new_node)
        return has_variables

    @classmethod
    def plot(cls, x_axis_train,
             y_axis_train,
             y_axis_fitted,
             x_axis_test=None,
             y_axis_test=None,
             y_axis_pred=None,
             test_set=False):
        """Plots training and testing predicted graphs.

        This method plots training and testing predicted graphs.

        Args:
            x_axis_train (list): Training set `x` original data values
            y_axis_train (list): Training set `y` original data values
            y_axis_fitted (list): Training set `y` predicted values
            x_axis_test (list): Testing set `x` original data values
            y_axis_test (list): Testing set `y` original data values
            y_axis_pred (list): Testing set `y` predicted values
            test_set (bool): Controlling modifier of testing set
        """
        # adding additional point to remove the spae between train and test sets
        if test_set:
            x_axis_train = np.append(x_axis_train, x_axis_test[0])
            y_axis_train = np.append(y_axis_train, y_axis_test[0])
            y_axis_fitted = np.append(y_axis_fitted, y_axis_pred[0])

        ax = plt.axes()
        ax.grid(linestyle=':', linewidth=1, alpha=1, zorder=1)
        plt.xlabel("X")
        plt.ylabel("Y")
        line = [None, None, None, None]
        line[0], = ax.plot(x_axis_train, y_axis_train, linestyle='-',
                           color='black', linewidth=0.7, zorder=2, label='Targeted')
        line[1], = ax.plot(x_axis_train, y_axis_fitted, linestyle=':', color='red', marker='o',
                           markersize=3, markerfacecolor='white', linewidth=0.7, zorder=3, label='Generated')
        if test_set:
            line[2], = ax.plot(x_axis_test, y_axis_test, linestyle='-',
                               color='black', linewidth=0.5, zorder=2)
            line[3], = ax.plot(x_axis_test, y_axis_pred, linestyle=':', color='blue', marker='o',
                               markerfacecolor='white', markersize=3, linewidth=0.7, zorder=3, label='Generated')
            plt.axvline(x=x_axis_test[0], linestyle='-',
                        color='black', linewidth='1')
        plt.draw()
        plt.legend()
        plt.show()
