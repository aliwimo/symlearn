from graphviz import Digraph, Source
from random import random, randint, uniform
from statistics import mean
from copy import deepcopy
import numpy as np
from node import Node
from parameters import Par

def Add(X0, X1):
    return X0 + X1

def Sub(X0, X1):
    return X0 - X1

def Mul(X0, X1):
    return X0 * X1

def Div(X0, X1):
    sign_X1 = np.sign(X1)   # numpy.sign: X>=1 returns 1, X==0 returns 0, X<=1 returns -1 
    sign_X1[sign_X1 == 0] = 1
    return X0 / sign_X1

def Sin(X0):
    return np.sin(X0)

def Cos(X0):
    return np.cos(X0)

def Rlog(X0):
    return np.log(np.abs(X0) + 1e-6)

def Exp(X0):
    return np.exp(X0)

class Tree:
    operator_rate = 0.5
    terminal_rate = 0.5
    variable_rate = 0.5
    sharing_rate = 0.9

    def __init__(self, left=None, right=None, parent_id=None):
        self.root = Node(parent_id=parent_id)
        self.left = left
        self.right = right
        self.error = 10e6

    # create randomly a tree with 'grow' or 'full' methods
    def create_tree(self, method, min_depth, max_depth, current_depth=0):
        if method == 'full':
            if current_depth < max_depth - 1:
                if random() > Tree.operator_rate:
                    self.root.set_value("function")
                else:
                    self.root.set_value("operator")
            else:
                if random() > Tree.variable_rate:
                    self.root.set_value("variable")
                else:
                    self.root.set_value("constant")
                    
        elif method == 'grow':
            if current_depth < min_depth:
                if random() > Tree.operator_rate:
                    self.root.set_value("function")
                else:
                    self.root.set_value("operator")
            elif min_depth <= current_depth < max_depth - 1:
                if random() > Tree.terminal_rate:
                    if random() > Tree.operator_rate:
                        self.root.set_value("function")
                    else:
                        self.root.set_value("operator")
                else:
                    if random() > Tree.variable_rate:
                        self.root.set_value("variable")
                    else:
                        self.root.set_value("constant")
            else:
                if random() > Tree.variable_rate:
                    self.root.set_value("variable")
                else:
                    self.root.set_value("constant")

        # create left and right branches
        if self.root.value in Par.OPERATORS:
            self.left = Tree(parent_id=self.root.id)
            self.left.create_tree(method, min_depth, max_depth, current_depth + 1)
        if self.root.value in Par.OPERATORS or self.root.value in Par.FUNCTIONS:
            self.right = Tree(parent_id=self.root.id)
            self.right.create_tree(method, min_depth, max_depth, current_depth + 1)

    # print the tree as an equation
    def tree_equation(self):
        equation = "("
        if self.left: equation += self.left.tree_equation()
        equation += str(self.root.value)
        if self.right: equation += self.right.tree_equation()
        equation += ")"
        return equation

    # returns nodes of the tree
    def get_nodes(self, nodes_id=False):
        nodes = []
        nodes.append(self.root)
        if self.left: nodes = nodes + self.left.get_nodes()
        if self.right: nodes = nodes + self.right.get_nodes()
        if nodes_id: return [x.id for x in nodes]
        return nodes

    # returns depth of the tree
    def tree_depth(self):
        depth_left = 0
        depth_right = 0
        if self.left: depth_left = self.left.tree_depth()
        if self.right: depth_right = self.right.tree_depth()
        return max(depth_left, depth_right) + 1

    def random_node(self, node_type=None):
        nodes = self.get_nodes()
        selected_node = None
        correct_selected = False

        if node_type == "function":
            has_non_terminals = False
            for i in range(len(nodes)):
                if nodes[i].node_type != "variable" or nodes[i].node_type != "constant":
                    has_non_terminals = True
                    break
            if has_non_terminals:
                while not correct_selected:
                    selected_node = nodes[randint(0, len(nodes) - 1)]
                    if selected_node.node_type != "variable" or selected_node.node_type != "constant":
                        correct_selected = True
            else:
                selected_node = nodes[randint(0, len(nodes) - 1)]
        elif node_type == "variable" or node_type == 'constant':
            while not correct_selected:
                selected_node = nodes[randint(0, len(nodes) - 1)]
                if selected_node.node_type == "variable" or selected_node.node_type == "constant":
                    correct_selected = True
        else:
            selected_node = nodes[randint(0, len(nodes) - 1)]
        return selected_node

    def copy_subtree(self):
        selected_node = None
        if random() < Tree.sharing_rate:
            selected_node = self.random_node(node_type="function")
        else:
            selected_node = self.random_node(node_type="terminal")
        share_point = selected_node.id
        return self.find_tree(share_point, "copy")    

    def paste_subtree(self, subtree):
        selected_node = None
        if subtree.root.node_type != "variable" or subtree.root.node_type != "constant":
            selected_node = self.random_node(node_type="function")
        else:
            selected_node = self.random_node(node_type="terminal")
        share_point = selected_node.id
        self.find_tree(share_point, "paste", subtree)

    # takes an instance of a tree as a subtree
    def take_instance(self):
        sub = Tree()
        sub.root.value = self.root.value
        sub.root.node_type = self.root.node_type
        if self.left:  sub.left  = self.left.take_instance()
        if self.right: sub.right = self.right.take_instance()
        return sub

    # glues the subtree into this root
    def glue_instance(self, subtree):
        parent_id = self.root.parent_id
        self.root.value = subtree.root.value
        self.root.node_type = subtree.root.node_type
        self.root.parent_id = parent_id
        if subtree.left:
            self.left = Tree()
            self.left.glue_instance(subtree.left)
        else: self.left = None

        if subtree.right:
            self.right = Tree()
            self.right.glue_instance(subtree.right)
        else: self.right = None

    # change value of one node
    def change_subtree(self):
        rand = randint(1, 4)
        if rand == 1:
            node_type = "operator"
        elif rand == 2:
            node_type = "function"
        elif rand == 3:
            node_type = "variable"
        else: 
            node_type = "constant"

        method = 'grow' if randint(1, 2) == 1 else 'full'

        type_before = self.root.node_type
        self.root.set_value(node_type)

        if type_before == "operator":
            if rand != 1: self.left = None
            if rand == 3 or rand == 4: self.right = None            
        elif type_before == "function":
            if rand == 1: 
                self.left = Tree(parent_id=self.root.id)
                self.left.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
            if rand == 3 or rand == 4:
                self.left = None
                self.right = None
        else:
            if rand == 1:
                self.left = Tree(parent_id=self.root.id)
                self.left.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
            if rand == 1 or rand == 2:
                self.right = Tree(parent_id=self.root.id)
                self.right.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)

    def change_node(self):
        selected_node = self.random_node()
        self.find_tree(selected_node.id, "change")

    # search for a tree where its root id number equal root_id
    def find_tree(self, root_id, process, subtree=None):
        tree = None
        if root_id == self.root.id:
            if process == 'copy': tree = self.take_instance()
            elif process == 'paste': self.glue_instance(subtree)
            elif process == 'change': self.change_subtree()
        else:
            if self.left and root_id in self.left.get_nodes(nodes_id=True):
                tree = self.left.find_tree(root_id, process, subtree)
            else:
                tree = self.right.find_tree(root_id, process, subtree)
        return tree
  
    # returns the value of the tree using x and y (if exist) variables 
    def calc_tree(self, X):
        result = 1e30
        if self.root.node_type == "operator":
            if self.root.value == '+':
                result = Add(self.left.calc_tree(X), self.right.calc_tree(X))
            elif self.root.value == '-':
                result = Sub(self.left.calc_tree(X), self.right.calc_tree(X))
            elif self.root.value == '*':
                result = Mul(self.left.calc_tree(X), self.right.calc_tree(X))
            elif self.root.value == '/':
                result = Div(self.left.calc_tree(X), self.right.calc_tree(X))
        elif self.root.node_type == "function":
            if self.root.value == 'sin':
                result = Sin(self.right.calc_tree(X))
            elif self.root.value == 'cos':
                result = Cos(self.right.calc_tree(X))
            elif self.root.value == 'exp':
                result = Exp(self.right.calc_tree(X))
            elif self.root.value == 'rlog':
                result = Rlog(self.right.calc_tree(X))
        elif self.root.node_type == "variable":
            result = X[:, self.root.rank]
        else:
            result = np.array([self.root.value] * X.shape[0])
        return result

    def update_error(self):
        self.error = np.sum(np.abs(self.calc_tree(Par.X) - Par.Y))

    def simplify_tree(self):
        if self.left: self.left.simplify_tree()
        if self.right: self.right.simplify_tree()
        if self.root.value in Par.OPERATORS:
            result = self.perform_simplification()
            if result != None:
                self.root.value = str(result)
                self.left = None
                self.right = None

    def perform_simplification(self):
        try:
            if self.root.value == '+':
                result = int(self.left.root.value) + int(self.right.root.value)
            elif self.root.value == '-':
                result = int(self.left.root.value) - int(self.right.root.value)
            elif self.root.value == '*':
                result = int(self.left.root.value) * int(self.right.root.value)
            elif self.root.value == '/':
                result = int(self.left.root.value) * int(self.right.root.value)
        except:
            result = None
        return result

    # # pruning tree if max depth is exceeded without creating a new one
    # def prun_tree(self, max_depth, current_depth=1):
    #     if self.left: self.left.prun_tree(max_depth, current_depth=current_depth + 1)
    #     if self.right: self.right.prun_tree(max_depth, current_depth=current_depth + 1)
    #     if current_depth == max_depth:
    #         self.root.set_value(Par.TERMINALS, "terminal")
    #         self.left = None
    #         self.right = None

    # fix parent_id numbers
    def fix_tree(self, parent_id=None):
        self.root.parent_id = parent_id
        if self.left: self.left.fix_tree(parent_id=self.root.id)
        if self.right: self.right.fix_tree(parent_id=self.root.id)

    # export tree's graph
    def draw_tree(self, file_name, label):
        self.fix_tree()
        graph = [Digraph()]
        graph[0].attr(kw = 'graph', label = label)
        nodes = self.get_nodes()
        for n in nodes:
            n.draw(graph)
        Source(graph[0], filename = file_name + '.gv', format='png').render()

