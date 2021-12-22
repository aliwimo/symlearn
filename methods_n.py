from random import random, choice
from graphviz import Digraph, Source
from node_n import Node
import numpy as np

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
        instance_node = choice(source_nodes).subtree()
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
    def rank_trees(cls, trees, fitnesses, is_reversed=False):
        sorted_indices = np.argsort(fitnesses)
        if not is_reversed: sorted_indices = np.flip(sorted_indices)
        fitnesses.sort(reverse=not is_reversed)
        temp_trees = trees.copy()
        for (m, n) in zip(range(len(trees)), sorted_indices):
            trees[m] = temp_trees[n]
        return trees, fitnesses

    # @classmethod
    # def get_nodes(cls, root: Node):
    #     nodes = []
    #     nodes.append(root)
    #     if root.inputs == 2:
    #         nodes = nodes + cls.get_nodes(root.left)
    #         nodes = nodes + cls.get_nodes(root.right)
    #     elif root.inputs == 1:
    #         nodes = nodes + cls.get_nodes(root.right)
    #     return nodes

    @classmethod
    def export_graph(cls, root: Node, file_name, label):
        graph = [Digraph()]
        graph[0].attr(kw = 'graph', label = label)
        cls.draw_node(graph, root)
        Source(graph[0], filename = file_name + '.gv', format='png').render()

    @classmethod
    def draw_node(cls, graph, root: Node):
        graph[0].node(str(root.id), str(root))
        if root.left:
            graph[0].edge(str(root.id), str(root.left.id))
            cls.draw_node(graph, root.left)
        if root.right:
            graph[0].edge(str(root.id), str(root.right.id))
            cls.draw_node(graph, root.right)