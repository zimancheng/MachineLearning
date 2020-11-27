from svm_simple import *

class ModelStructure(object):
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
        self.alpha = np.zeros(self.m)
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
    valid_err_ind = [ind for ind in range(ds.m) if ds.err[ind, 0] > 0] 

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
        best_j = select_random_j(i, ds.m)
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
            non_bound_ind = [ind for ind in range(ds.m) if ds.alpha[ind] > 0 and ds.alpha[ind] < ds.C]

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

def inner_loop(i, ds):
    """Inner loop to find and update alpha j.
    
    Args:
        i: index of the first alpha in pair
        ds: the data structure of SVM model
    Returns:
        0 if no alpha pairs changed. 1 if a pair of alpha changed.
    """
    err_i = calculate_error(ds, i)

    if (ds.labels[i] * err_i < -ds.tolerance and ds.alpha[i] < ds.C) or \
        (ds.labels[i] * err_i > ds.tolerance and ds.alpha[i] > 0):
        j, err_j = select_j(ds, i, err_i)

        alpha_i_pre = np.copy(ds.alpha[i])
        alpha_j_pre = np.copy(ds.alpha[j])

        if ds.labels[i] == ds.labels[j]:
            L = max(0, ds.alpha[j] + ds.alpha[i] - ds.C)
            H = min(ds.C, ds.alpha[j] + ds.alpha[i])
        else:
            L = max(0, ds.alpha[j] - ds.alpha[i])
            H = min(ds.C, ds.C + ds.alpha[j] - ds.alpha[i])
        if L == H: print("L==H"); return 0
        
        eta = 2.0 * ds.dataset[i].dot(ds.dataset[j]) - ds.dataset[i].dot(ds.dataset[i]) - ds.dataset[j].dot(ds.dataset[j])
        if eta >= 0: print("eta >= 0"); return 0

        ds.alpha[j] -= ds.labels[j] * (err_i - err_j) / eta
        ds.alpha[j] = clip_alpha(ds.alpha[j], L, H)
        ds.err[j] = [1, err_j]

        if abs(ds.alpha[j] - alpha_j_pre) < 1e-5:
            print("alpha j is not moving enough")
            return 0

        ds.alpha[i] += ds.labels[i] * ds.labels[j] * (alpha_j_pre - ds.alpha[j])
        ds.err[i] = [1, err_i]

        b1 = ds.b - err_i - ds.labels[i] * (ds.alpha[i] - alpha_i_pre) * ds.dataset[i].dot(ds.dataset[i]) \
                          - ds.labels[j] * (ds.alpha[j] - alpha_j_pre) * ds.dataset[j].dot(ds.dataset[i])
        b2 = ds.b - err_j - ds.labels[i] * (ds.alpha[i] - alpha_i_pre) * ds.dataset[i].dot(ds.dataset[j]) \
                          - ds.labels[j] * (ds.alpha[j] - alpha_j_pre) * ds.dataset[j].dot(ds.dataset[j])
        
        if 0 < ds.alpha[i] and ds.alpha[i] < ds.C:
            ds.b = b1
        elif 0 < ds.alpha[j] and ds.alpha[j] < ds.C:
            ds.b = b2
        else:
            ds.b = (b1 + b2) / 2
        
        return 1
    else:
        return 0

if __name__ == "__main__":
    dataset, labels = load_dataset('testSet.txt')
    alpha, b = smo_full(dataset, labels, 0.6, 0.001, 200)
    plot_model(dataset, labels, alpha, b, 'svm_full.png')