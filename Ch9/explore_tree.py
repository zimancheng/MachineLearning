import regression_tree
import numpy as np
import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def plot_model(min_loss, min_n):
    """Plot the regression tree or model tree on given dataset.
    
    Args:
        min_loss: minimum loss tolerance of split
        min_n: minimum examples allowed in a treenode
    """
    plot_model.f.clf()
    plot_model.ax = plot_model.f.add_subplot(111)

    if button_var.get():
        if min_n < 2:
            min_n = 2
        
        tree = regression_tree.build_tree(plot_model.dataset, regression_tree.linear_leaf, 
                                            regression_tree.calculate_linear_model_error, (min_loss, min_n))
        y_pred = regression_tree.predict(tree, plot_model.dataset, 'model')
    else:
        tree = regression_tree.build_tree(plot_model.dataset, ops=(min_loss, min_n))
        y_pred = regression_tree.predict(tree, plot_model.dataset, 'CART')
    
    plot_model.ax.scatter(plot_model.dataset[:, 0], plot_model.dataset[:, 1], s=5)
    plot_model.ax.plot(plot_model.dataset[:, 0], y_pred, linedwidth=2.0) 
    plot_model.canvas.show()

def get_inputs():
    try:
        min_n = int(min_n_entry.get())
    except:
        min_n = 10
        print('Enter Integer for min_n')
        min_n_entry.delete(0, END)
    
    try:
        min_loss = float(min_loss_entry.get())
    except:
        min_loss = 1.0
        print('Enter Float for min_loss')
        min_loss_entry.delete(0, END)
        min_loss_entry.insert(0, '1.0')
    return min_n, min_loss

def draw_tree():
    """Obtain the parameters from input and draw tree."""
    min_n, min_loss = get_inputs()
    plot_model(min_loss, min_n)

window = tk.Tk()
# tk.Label(window, text='Plot Place Holder').grid(row=0, columnspan=3)
plot_model.f = Figure(figsize=(5, 4), dpi=100)
plot_model.canvas = FigureCanvasTkAgg(plot_model.f, master=window)
plot_model.canvas.show()
plot_model.canvas.get_tk_widget().grid(row=0, columnspan=3)

tk.Label(window, text='min loss').grid(row=1, column=0)
min_loss_entry = tk.Entry(window)
min_loss_entry.grid(row=1, column=1)
min_loss_entry.insert(0, '1.0')

tk.Label(window, text='min n').grid(row=2, column=0)
min_n_entry = tk.Entry(window)
min_n_entry.grid(row=2, column=1)
min_n_entry.insert(0, '10')

tk.Button(window, text='Re-draw', command=draw_tree).grid(row=1, column=2, rowspan=3)

button_var = tk.IntVar()
check_button = tk.Checkbutton(window, text='Model Tree', variable=button_var)
check_button.grid(row=3, column=0, columnspan=2)

plot_model.dataset = regression_tree.load_dataset('./data/sine.txt')
plot_model(1.0, 10)

tk.mainloop()