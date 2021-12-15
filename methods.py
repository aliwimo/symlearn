from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
from tree import Tree


class Methods:

    # create initial trees by ramped half & half method
    @classmethod
    def init_trees(cls, pop_size, initial_min_depth, initial_max_depth):
        pop = []
        for _ in range(pop_size // 2):
            tree = Tree() 
            tree.create_tree('full', initial_min_depth, initial_max_depth)
            pop.append(tree)
        for _ in range((pop_size // 2), pop_size):
            tree = Tree() 
            tree.create_tree('grow', initial_min_depth, initial_max_depth)
            pop.append(tree)
        return pop

    @classmethod
    def rank_trees(cls, trees, errors, is_reversed=False):
        sorted_indices = np.argsort(errors)
        if not is_reversed: sorted_indices = np.flip(sorted_indices)
        errors.sort(reverse=not is_reversed)
        temp_trees = trees.copy()
        for (m, n) in zip(range(len(trees)), sorted_indices):
            trees[m] = temp_trees[n]
        return trees, errors
    
    @classmethod
    def share(cls, tree1, tree2):
        sub = tree1.copy_subtree()
        tree2.paste_subtree(sub)

    @classmethod
    def check_depth(cls, tree: Tree, initial_min_depth, initial_max_depth, max_depth):
        if tree.tree_depth() > max_depth:
            method = 'grow' if randint(1, 2) == 1 else 'full'
            tree.create_tree(method, initial_min_depth, initial_max_depth)
        return tree

    @classmethod
    def plot(self, X_train, X_test, y_train, y_test, y_model, y_predict):
        # preparing plot
        ax = plt.axes()
        # showing grid 
        ax.grid(linestyle=':', linewidth=1, alpha=1, zorder=0)
        # set the Label of X and Y axis
        plt.xlabel("X")
        plt.ylabel("Y")
        # for markers and colors look ar the end of this file
        line = [None, None, None, None]
        line[0], = ax.plot(X_train[:, 0], y_train, linestyle='-', color='black', linewidth=0.5, zorder=1)    
        line[1], = ax.plot(X_test[:, 0], y_test, linestyle='-', color='black', linewidth=0.5, zorder=1)
        line[2], = ax.plot(X_train[:, 0], y_model, linestyle=':', color='red', linewidth=2, zorder=2)
        line[3], = ax.plot(X_test[:, 0], y_predict, linestyle=':' ,color='green', linewidth=2, zorder=2)
        # show graphes
        plt.draw()
        plt.show()

    