from parameters import Parameters
from expressions import *
from graphviz import Digraph, Source
from random import random, choice
from copy import deepcopy

class Functions():

    @classmethod
    def generate_population(cls):
        population = []
        # generate with full method
        for _ in range(Parameters.POP_SIZE // 2):
            individual = cls.generate_individual('full')
            population.append(individual)
        # generate with grow method
        for _ in range((Parameters.POP_SIZE // 2), Parameters.POP_SIZE):
            individual = cls.generate_individual('grow')
            population.append(individual)
        return population


    @classmethod
    def generate_individual(cls, method, current_depth=0) -> Node:
        if method == 'full':
            if current_depth < Parameters.INIT_MAX_DEPTH - 1:
                node = choice(Parameters.EXPRESSIONS)()
            else:
                node = choice(Parameters.TERMINALS)()
        elif method == 'grow':
            if current_depth < Parameters.INIT_MIN_DEPTH:
                node = choice(Parameters.EXPRESSIONS)()
            elif Parameters.INIT_MIN_DEPTH <= current_depth < Parameters.INIT_MAX_DEPTH - 1:
                if random() > 0.5:
                    node = choice(Parameters.EXPRESSIONS)()
                else:
                    node = choice(Parameters.TERMINALS)()
            else:
                node = choice(Parameters.TERMINALS)()
        # create left and right branches
        if node.inputs == 2:
            child_1 = cls.generate_individual(method, current_depth + 1)
            child_2 = cls.generate_individual(method, current_depth + 1)
            node.add_left_child(child_1)
            node.add_right_child(child_2)
        elif node.inputs == 1:
            child = cls.generate_individual(method, current_depth + 1)
            node.add_right_child(child)
        return node

    
    @classmethod
    def export_graph(cls, root: Node, file_name, label):
        graph = [Digraph()]
        graph[0].attr(kw = 'graph', label = label)
        root.draw_node(graph)
        Source(graph[0], filename = file_name + '.gv', format='png').render()

    
    @classmethod
    def share(cls, source: Node, target: Node):
        source_nodes = cls.get_nodes(source)
        instance_node = deepcopy(choice(source_nodes))
        target_nodes = cls.get_nodes(target)
        removed_node = choice(target_nodes)
        parent = removed_node.parent
        if parent:
            if removed_node.position == 'left':
                parent.add_left_child(instance_node)
            elif removed_node.position == 'right':
                parent.add_right_child(instance_node)
            return target
        else:
            return instance_node



    @classmethod
    def get_nodes(cls, root: Node):
        nodes = []
        nodes.append(root)
        if root.inputs == 2:
            nodes = nodes + cls.get_nodes(root.left)
            nodes = nodes + cls.get_nodes(root.right)
        elif root.inputs == 1:
            nodes = nodes + cls.get_nodes(root.right)
        return nodes

