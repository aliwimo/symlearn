from parameters import Parameters
from expressions import *
from graphviz import Digraph, Source
from random import random, choice
from os import remove

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
        Source(graph[0], filename = file_name, format='png').render()
        remove(file_name)
