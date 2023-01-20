import numpy as np
from copy import deepcopy
from random import random, choice
from symlearn.core.node import Node
from symlearn.core.functions import Constant

def share(source: Node, target: Node):
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

def substitute(source: Node, nodes_pool):
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

def simplify(root: Node):
    """Simplifies tree by combining nodes.

    This method simplifies trees by combining tree's branches
    that does not have any variable nodes.

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
        has_variables = simplify(root.left)
    if root.right:
        has_variables = simplify(root.right)
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