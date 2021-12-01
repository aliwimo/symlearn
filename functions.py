from random import randint, random
import numpy as np
from parameters import Par
from tree import Tree


class Func:

    # dataset of targeted function
    @classmethod
    def generate_dataset(cls):
        dataset = []
        points_x = np.linspace(Par.DOMAIN_X[0], Par.DOMAIN_X[1], Par.POINT_NUM)
        if Par.VAR_NUM == 1:

            for i in range(Par.POINT_NUM):
                dataset.append([points_x[i], Par.TARGET_FUNC(points_x[i])])
        else:
            points_y = np.linspace(Par.DOMAIN_Y[0], Par.DOMAIN_Y[1], Par.POINT_NUM)
            for i in range(Par.POINT_NUM):
                dataset.append([points_x[i], points_y[i], Par.TARGET_FUNC(points_x[i], points_y[i])])
        return dataset

    # dataset of generated function
    @classmethod
    def computed_dataset(cls, tree):
        dataset = []
        points_x = np.linspace(Par.DOMAIN_X[0], Par.DOMAIN_X[1], Par.POINT_NUM)
        if Par.VAR_NUM == 1:
            for i in range(Par.POINT_NUM):
                dataset.append([points_x[i], tree.calc_tree(points_x[i])])
        else:
            points_y = np.linspace(Par.DOMAIN_Y[0], Par.DOMAIN_Y[1], Par.POINT_NUM)
            for i in range(Par.POINT_NUM):
                dataset.append([points_x[i], points_y[i], tree.calc_tree(points_x[i], points_y[i])])
        return dataset

    # create initial trees by ramped half & half method
    @classmethod
    def init_trees(cls):
        pop = []
        for _ in range(Par.POP_SIZE // 2):
            tree = Tree() 
            tree.create_tree('full', Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
            if len(pop) > 1:
                is_different = cls.control_difference(pop, tree)
                while not is_different:
                    tree.create_tree('full', Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
                    is_different = cls.control_difference(pop, tree)
            pop.append(tree)
        for _ in range((Par.POP_SIZE // 2), Par.POP_SIZE):
            tree = Tree() 
            tree.create_tree('grow', Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
            is_different = cls.control_difference(pop, tree)
            while not is_different:
                tree.create_tree('full', Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
                is_different = cls.control_difference(pop, tree)
            pop.append(tree)
        return pop

    @classmethod
    def rank_trees(cls, trees, errors, is_reversed=False):
        sorted_indices = np.argsort(errors)
        if not is_reversed: sorted_indices = np.flip(sorted_indices)
        errors.sort(reverse=not is_reversed)
        temp_trees = trees.copy()
        for (m, n) in zip(range(Par.POP_SIZE), sorted_indices):
            trees[m] = temp_trees[n]
        return trees, errors
    
    @classmethod
    def share(cls, tree1, tree2):
        sub = tree1.copy_subtree()
        tree2.paste_subtree(sub)

    @classmethod
    def check_depth(cls, tree: Tree):
        if tree.tree_depth() > Par.MAX_DEPTH:
            method = 'grow' if randint(1, 2) == 1 else 'full'
            tree.create_tree(method, Par.INIT_MIN_DEPTH, Par.INIT_MAX_DEPTH)
        return tree
            
    @classmethod
    def control_difference(cls, pop, tree: Tree):
        different = True
        for i in range(len(pop)):
            different = cls.is_different(pop[i], tree)
            if not different:
                break
        return different 
    
    @classmethod
    def is_different(cls, t1: Tree, t2: Tree):
        different = False
        if t1 and t2:
            different = cls.is_different(t1.left, t2.left)
            if t1.root.value != t2.root.value:
                different = True
                return different
            if not different:
                different = cls.is_different(t1.right, t2.right)
        elif (t1 and not t2) or (not t1 and t2):
            different = True
        return different

    @classmethod
    def progress_bar(cls, iteration, gen, error, total, other, length=100):
        fill = '█'
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r Progress: |{bar}| {percent}% Complete | Counter: {iteration} | Gen: {gen} | Error: {error}', end = '\r')
        # Print New Line on Complete 
        if iteration == total: 
            print()