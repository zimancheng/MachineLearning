import trees
import tree_plotter

def test():
    fr = open("lenses.txt")
    lense_data = [example.strip().split('\t') for example in fr.readlines()] 
    ftr_list = ['age', 'prescript', 'astigmatic', 'tearRate']

    lense_tree = trees.build_tree(lense_data, ftr_list)
    print(lense_tree)

    tree_plotter.create_plot(lense_tree)
