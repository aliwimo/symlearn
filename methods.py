import numpy as np
import matplotlib.pyplot as plt
from random import random, choice
from parameters import Parameters
from graphviz import Digraph, Source
from node import Node
from copy import deepcopy
from functions import Constant

class Methods:

    @classmethod
    def generate_population(cls, pop_size, initial_min_depth, initial_max_depth, expressions, terminals):
        population = []
        # generate with full method
        for _ in range(pop_size // 2):
            individual = cls.generate_individual('full', initial_min_depth, initial_max_depth, expressions, terminals)
            population.append(individual)
        # generate with grow method
        for _ in range((pop_size // 2), pop_size):
            individual = cls.generate_individual('grow', initial_min_depth, initial_max_depth, expressions, terminals)
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
        if node.inputs == 2:
            child_1 = cls.generate_individual(method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            child_2 = cls.generate_individual(method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            node.add_left_node(child_1)
            node.add_right_node(child_2)
        elif node.inputs == 1:
            child = cls.generate_individual(method, initial_min_depth, initial_max_depth, expressions, terminals, current_depth + 1)
            node.add_right_node(child)
        return node

    @classmethod
    def share(cls, source: Node, target: Node):
        source_nodes = source.sub_nodes()
        
        if len(source_nodes) > 1:
            if random() < 0.9:
                is_function = False
                while not is_function:
                    instance_node = choice(source_nodes).sub_tree()
                    if instance_node.inputs >= 1:
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
        source_nodes = source.sub_nodes()
        selected_node = choice(source_nodes)
        same_inputs = False
        while not same_inputs:
            new_node = choice(nodes_pool)()
            if new_node.inputs == selected_node.inputs:
                same_inputs = True
        parent = selected_node.parent
        if parent:
            if selected_node.parent.left == selected_node:
                parent.remove_left_node(selected_node)
                parent.add_left_node(new_node)
            elif selected_node.parent.right == selected_node:
                parent.remove_right_node(selected_node)
                parent.add_right_node(new_node)

        if selected_node.inputs == 2:
            new_node.add_left_node(deepcopy(selected_node.left))
            new_node.add_right_node(deepcopy(selected_node.right))
        elif selected_node.inputs == 1:
            new_node.add_right_node(deepcopy(selected_node.right))
        return source
    
    @classmethod
    def rank_trees(cls, trees, fitnesses, is_reversed=False):
        sorted_indices = np.argsort(fitnesses)
        if not is_reversed: sorted_indices = np.flip(sorted_indices)
        fitnesses.sort(reverse=not is_reversed)
        temp_trees = trees.copy()
        for (m, n) in zip(range(len(trees)), sorted_indices):
            trees[m] = temp_trees[n]
        return trees, fitnesses

    @classmethod
    def export_graph(cls, root: Node, file_name, label):
        graph = [Digraph()]
        graph[0].attr(kw = 'graph', label = label)
        cls.draw_node(graph, root)
        Source(graph[0], filename = file_name + '.gv', format=Parameters.EXPORT_EXT).render()

    @classmethod
    def draw_node(cls, graph, root: Node):
        graph[0].node(str(root.id), str(root))
        if root.left:
            graph[0].edge(str(root.id), str(root.left.id))
            cls.draw_node(graph, root.left)
        if root.right:
            graph[0].edge(str(root.id), str(root.right.id))
            cls.draw_node(graph, root.right)


    @classmethod
    def simplify(cls, root: Node):
        has_variables = False
        if root.type == 'variable': has_variables = True
        if root.left: has_variables = cls.simplify(root.left)
        if root.right: has_variables = cls.simplify(root.right)
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
    def plot(cls, x_axis_train, y_axis_train, y_axis_fitted, x_axis_test=None, y_axis_test=None, y_axis_pred=None, test_set=False):

        # adding additional point to remove the spae between train and test sets
        if test_set:
            x_axis_train = np.append(x_axis_train, x_axis_test[0])
            y_axis_train = np.append(y_axis_train, y_axis_test[0])
            y_axis_fitted = np.append(y_axis_fitted, y_axis_pred[0])

        # preparing plot
        ax = plt.axes()
        # showing grid 
        ax.grid(linestyle=':', linewidth=1, alpha=1, zorder=1)
        # set the Label of X and Y axis
        plt.xlabel("X")
        plt.ylabel("Y")
        # for markers and colors look ar the end of this file
        line = [None, None, None, None]
        line[0], = ax.plot(x_axis_train, y_axis_train, linestyle='-', color='black', linewidth=0.7, zorder=2, label='Targeted')    
        line[1], = ax.plot(x_axis_train, y_axis_fitted, linestyle=':', color='red', marker='o', markersize=3, markerfacecolor='white', linewidth=0.7, zorder=3, label='Generated')
        # line[1], = ax.plot(x_axis_train, y_axis_fitted, linestyle=':', color='black', linewidth=0.7, zorder=3, label='Generated')
        if test_set:
            line[2], = ax.plot(x_axis_test, y_axis_test, linestyle='-', color='black', linewidth=0.5, zorder=2)
            line[3], = ax.plot(x_axis_test, y_axis_pred, linestyle=':', color='blue', marker='o', markerfacecolor='white', markersize=3, linewidth=0.7, zorder=3, label='Generated')
            # line[3], = ax.plot(x_axis_test, y_axis_pred, linestyle=':' ,color='black', linewidth=0.7, zorder=3)
            plt.axvline(x=x_axis_test[0], linestyle='-', color='black', linewidth='1')
        # show graphes
        plt.draw()
        plt.legend()
        plt.show()
