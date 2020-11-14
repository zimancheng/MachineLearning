#!/usr/bin/env python

# -*- coding: utf-8 -*

import matplotlib.pyplot as plt 

decision_node = dict(boxstyle = "sawtooth", fc = "0.8")
leaf_node = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

def plot_node(node_txt, paren_pos, node_pos, node_type):
    """
        Function:plot_node
        Plot a node of a tree on an existing axe, with an arrow from parent node to it
        :params:node_txt - feature name str 
        :params:paren_pos - the parent node coordinates
        :params:node_pos - current node coordinates
        :params:node_type - leaf or node
        :return:none
    """
    create_plot.ax1.annotate(node_txt, xy = paren_pos, xycoords = "axes fraction",
                             xytext = node_pos, textcoords = "axes fraction", 
                             va = "center", ha = "center", bbox = node_type, arrowprops = arrow_args)
    
def create_plot(tree):
    """
        Function:create_plot
        Create a tree plot and show it
        :params:input none
        :return:none
    """
    fig = plt.figure(1, facecolor = "white") #number of this figure is 1
    fig.clf() #clear figure
    axprops = dict(xticks = [], yticks = [])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops) # frame is not on

    plot_tree.total_leaves = float(get_leaf_num(tree))
    plot_tree.total_depth = float(get_tree_depth(tree))

    plot_tree.x_coord = -0.5/plot_tree.total_leaves
    plot_tree.y_coord = 1.0

    plot_tree(tree, (0.5, 1.0), '')
    plt.show()

def get_leaf_num(tree):
    """
        Function:get_leaf_num
        Get the number of leaf nodes so that we can properly place nodes in X axis
        :params:tree - a nested dict
        :return:integer as number of leaf nodes
    """
    num_leaves = 0
    fir_key = list(tree)[0]
    subtree = tree[fir_key]

    for key in subtree.keys():
        if type(subtree[key]).__name__ == 'dict':
            num_leaves += get_leaf_num(subtree[key])
        else: num_leaves += 1
    return num_leaves   

def get_tree_depth(tree):
    """
        Function:get_tree_depth
        Get the depth of the tree, if it's 0 -> 1 -> 2, then it's depth should be 2
        :params:tree - a nested dict
        :return:an integer
    """
    max_tree_depth = 0
    fir_key = list(tree)[0]
    subtree = tree[fir_key]

    for key in subtree.keys():
        if type(subtree[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(subtree[key])
        else: this_depth = 1
        if max_tree_depth < this_depth:
            max_tree_depth = this_depth
    return max_tree_depth

def create_sample_tree(i):
    """
        Function:create_sample_tree
        Contains two sample trees in a list for testing the other functions in this module
        :params:i - the index of the tree in the list
        :return:a nested dict as tree
    """
    tree_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, 
                 {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 
                                                             1: 'no'}}}}]
    return tree_list[i]

def plot_arrow_txt(node_pos, paren_pos, txt_str):
    """
        Function:plot_arrow_txt
        Plot the text in the middle of an arrow to indicate the feature value the branch splits on
        :params:node_pos - the coordinates of the node pointed to
        :params:paren_pos - the coordinates of the parant node, pointing from which
        :params:txt_str - the feature value as a string
        :return:none
    """  
    x_mid = 0.5*(paren_pos[0] - node_pos[0]) + node_pos[0]
    y_mid = 0.5*(paren_pos[1] - node_pos[1]) + node_pos[1]
    create_plot.ax1.text(x_mid, y_mid, txt_str, va = "center", ha = "center", rotation = 30)

def plot_tree(tree, paren_pos, paren_val):
    """
        Function:plot_tree
        Recursively plot the tree.
            1. Each tree is a nested dict e.g. {'a': {0: 'yes', 1: {'b': {0: 'no', 1: 'yes}}} 
            2. Plot the decision node of the tree first, i.e. plot the root ('a' node) of this tree
            3. Then for each of its subtrees, if it's a leaf, then plot the leaf and update xoff.
            4. If it's not a leaf, call plot_tree again to plot this subtree. 
        Take every input tree's first key as a decision node since if it's not, it would been plotted 
        in the previous layer.

        :params:tree - a nested dict
        :params:paren_pos - the parent node coordinates of the input tree
        :params:paren_val - the feature value of the parent node
        :return:none
    """
    
    num_leaves = get_leaf_num(tree)     # get the number of leaves in the tree
    # tree_depth = get_tree_depth(tree) the depth of this tree will not be used
    node_pos = (plot_tree.x_coord + (num_leaves + 1.0)/(2*plot_tree.total_leaves), 
                plot_tree.y_coord)      # calculate the coordinates of node
    
    fir_ftr = list(tree)[0]
    plot_node(fir_ftr, paren_pos, node_pos, decision_node)      # plot the decision node
    plot_arrow_txt(node_pos, paren_pos, paren_val)     

    subtree = tree[fir_ftr] 
    plot_tree.y_coord -= 1/plot_tree.total_depth    # plot one layer down

    for key in subtree.keys():
        if type(subtree[key]).__name__ == 'dict':
            plot_tree(subtree[key], node_pos, str(key)) # the key value has to be casted to string
        else:
            plot_tree.x_coord = plot_tree.x_coord + 1/plot_tree.total_leaves
            plot_node(subtree[key], node_pos, (plot_tree.x_coord, plot_tree.y_coord), leaf_node)
            plot_arrow_txt((plot_tree.x_coord, plot_tree.y_coord), node_pos, str(key))
    
    plot_tree.y_coord += 1/plot_tree.total_depth
