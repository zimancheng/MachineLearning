from svm_simple import *

class ModelStructure(object):
"""Data Structure for storing parameters."""

    def __init__(self, dataset, labels, C, tolerance):
        """
        Args:
            dataset: ndarray of x matrix. Shape (m, n)
            labels: array of y labels. Shape (m,)
            C: slack parameter
            tolerance: error tolerance. floating point
        """
        self.dataset = dataset
        self.labels = labels
        self.C = C
        self.tolerance = tolerance
        self.m = dataset.shape[0]
        self.n = dataset.shape[1]
        self.alpha = np.zeros(m)
        self.b = 0
        self.err = np.zeros((self.m, 2))    # first column is a valid flag

def calculate_error(ds, k):
    """Calculate the error given alpha[k]
        
    Args:
        ds: the data structure of model
        k: kth alpha
    """
    err_k = (ds.alpha * ds.labels).dot(ds.dataset.dot(ds.dataset[k])) + ds.b - ds.labels[k]
    return err_k

def select_j(ds, i, err_i):
    """Given alpha[i], select alpha[j] that gives the maximum delta err
    
    Args:
        ds: the data structure of model
        i: the ith alpha in array
        err_i: the error calculated for alpha[i]
    Returns:
        best_j: j that renders the max delta error
        best_j_err: the error of the selected alpha j
    """
    # initialize j
    best_j = -1
    best_j_err = 0
    max_delta_err = 0
    ds.err[i] = [1, err_i]  # set valid 

    # take the index of alphas that has valid error cache
    valid_err_ind = [ind for ind in range(m) if ds.err[ind, 0] > 0] 

    # loop through alpha that gives max delta err
    if len(valid_err_ind) > 1:
        for j in valid_err_ind:
            if j == i: continue
            err_j = calculate_error(ds, j)
            delta_err = abs(err_i - err_j)

            if delta_err > max_delta_err:
                best_j = j
                max_delta_err = delta_err
                best_j_err = err_j
        return best_j, best_j_err
    else:   # no valid error cache values (first round)
        best_j = svm_simple.select_random_j(i, ds.m)
        best_j_err = calculate_error(ds, best_j)
    return best_j, best_j_err

def smo_full(dataset, labels, C, tolerance, max_iter):
    """Full Platt SMO
    
    Args:
        dataset: ndarray of x matrix. Shape (m, n)
        labels: array of y labels. Shape (m,)
        C: slack parameter
        tolerance: error tolerance. floating point
        max_iter: maximum iterations
    Returns:
        ds.alpha: the alpha parameter of model
        ds.b: the b paramter of model data structure
    """
    ds = ModelStructure(dataset, labels, C, tolerance)
    iter = 0
    full_scan = True
    alpha_pair_changed = 0

    # Select alpha i: from support vector or from whole dataset
    # first, scan the whole dataset to change alpha
    # after first round, optimize the alpha within interval[0, C], these are support vectors.
    # if for sup_vec, no alpha pairs changed, scan full data again to update alpha pairs.
    # stop until: 1. no alpha pairs need to update, then keep scanning full data until reach max_cycle
    #             2. just update alpha pairs until reach maximum cycle
    while iter < max_iter and (alpha_pair_changed > 0 or full_scan):
        alpha_pair_changed = 0

        if full_scan:
            for i in range(ds.m):
                alpha_pair_changed += inner_loop(i, ds)
                print("fullset iteration: {} i: {}, pairs changed: {}".format(iter, i, alpha_pair_changed))
        else:
            # select from support vectors, i.e. 0 < alpha i < C
            non_bound_ind = [ind for ind in range(ds.m) if ds.alpha > 0 and ds.alpha < C]

            for i in non_bound_ind:
                alpha_pair_changed += inner_loop(i, ds)
                print("non-bound iteration: {} i: {}, pairs changed: {}".format(iter, i, alpha_pair_changed))
            iter += 1

        if full_scan: 
            full_scan = False
        elif alpha_pair_changed == 0:
            full_scan = True
        print("iteration number: {}".format(iter))
    
    return ds.alpha, ds.b



