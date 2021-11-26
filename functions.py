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
            node.add_child(child_1)
            node.add_child(child_2)
        elif node.inputs == 1:
            child = cls.generate_individual(method, current_depth + 1)
            node.add_child(child)
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
        print(f'Instance node equation {instance_node.equation()}')
        target_nodes = cls.get_nodes(target)
        removed_node = deepcopy(choice(target_nodes))
        print(removed_node)
        terget_parent = removed_node.parent
        print(terget_parent)
        if terget_parent:
            index = terget_parent.children.index(removed_node)
            terget_parent.children.pop(index)
            terget_parent.children.insert(index, instance_node)
            instance_node.parent = terget_parent
            return target
        else:
            return instance_node

        # terget_parent.children[index] = deepcopy(instance_node)
        print(target.equation())



    @classmethod
    def get_nodes(cls, root: Node):
        nodes = []
        nodes.append(root)
        if root.inputs == 2:
            nodes = nodes + cls.get_nodes(root.children[0])
            nodes = nodes + cls.get_nodes(root.children[1])
        elif root.inputs == 1:
            nodes = nodes + cls.get_nodes(root.children[0])
        return nodes

