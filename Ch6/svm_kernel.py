from svm_full import *

def calculate_kernel(x_mat, x_i, kernel):
    """Calculate the distance or similarity between x[i] and all examples in x.
    
    Args:
        x_mat: ndarray of dataset. Shape (m, n)
        x_i: ith example of dataset. Array of shape (m,)
        kernel: tuple of kernel parameters. (str, float)
    Returns:
        k: result of <x, x_i>. Array of shape (m,)
    """
    m, n = x_mat.shape
    k = np.zeros(m)

    if kernel[0] == 'lin':  # linear kernel
        k = x_mat.dot(x_i)
    elif kernel[0] == 'rbf':    # radial bias function kernel
        delta = x_mat - x_i
        for i in range(m):
            k[i] = np.exp(-delta[i].dot(delta[i]) / kernel[1] ** 2)
    else:
        raise NameError('Unrecognizable kernel. Please use linear or radial biased kernel.')
    
    return k

class KernelSVM(ModelStructure):
    """Kernel SVM adds kernel parameter on ModelStructure."""
    def __init__(self, dataset, labels, C, tolerance, kernel):
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
        self.kernel = kernel
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = calculate_kernel(self.dataset, self.dataset[i, :], self.kernel)     # K[i, j] is the similarity between x_i and x_j


def calculate_error_ker(ds, k):
    """Calculate the error given alpha[k]
        
    Args:
        ds: the data structure of model
        k: kth alpha
    """
    err_k = (ds.alpha * ds.labels).dot(ds.K[:, k]) + ds.b - ds.labels[k]
    return err_k

def select_j_ker(ds, i, err_i):
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
            err_j = calculate_error_ker(ds, j)
            delta_err = abs(err_i - err_j)

            if delta_err > max_delta_err:
                best_j = j
                max_delta_err = delta_err
                best_j_err = err_j
        return best_j, best_j_err
    else:   # no valid error cache values (first round)
        best_j = select_random_j(i, ds.m)
        best_j_err = calculate_error_ker(ds, best_j)
    return best_j, best_j_err

def smo_kernel(dataset, labels, C, tolerance, kernel, max_iter):
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
    ds = KernelSVM(dataset=dataset, labels=labels, C=C, tolerance=tolerance, kernel=kernel)
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
                alpha_pair_changed += inner_loop_ker(i, ds)
                print("fullset iteration: {} i: {}, pairs changed: {}".format(iter, i, alpha_pair_changed))
        else:
            # select from support vectors, i.e. 0 < alpha i < C
            non_bound_ind = [ind for ind in range(ds.m) if ds.alpha[ind] > 0 and ds.alpha[ind] < ds.C]

            for i in non_bound_ind:
                alpha_pair_changed += inner_loop_ker(i, ds)
                print("non-bound iteration: {} i: {}, pairs changed: {}".format(iter, i, alpha_pair_changed))
            iter += 1

        if full_scan: 
            full_scan = False
        elif alpha_pair_changed == 0:
            full_scan = True
        print("iteration number: {}".format(iter))
    
    return ds.alpha, ds.b

def inner_loop_ker(i, ds):
    """Inner loop to find and update alpha j.
    
    Args:
        i: index of the first alpha in pair
        ds: the data structure of SVM model
    Returns:
        0 if no alpha pairs changed. 1 if a pair of alpha changed.
    """
    err_i = calculate_error_ker(ds, i)

    if (ds.labels[i] * err_i < -ds.tolerance and ds.alpha[i] < ds.C) or \
        (ds.labels[i] * err_i > ds.tolerance and ds.alpha[i] > 0):
        j, err_j = select_j_ker(ds, i, err_i)

        alpha_i_pre = np.copy(ds.alpha[i])
        alpha_j_pre = np.copy(ds.alpha[j])

        if ds.labels[i] == ds.labels[j]:
            L = max(0, ds.alpha[j] + ds.alpha[i] - ds.C)
            H = min(ds.C, ds.alpha[j] + ds.alpha[i])
        else:
            L = max(0, ds.alpha[j] - ds.alpha[i])
            H = min(ds.C, ds.C + ds.alpha[j] - ds.alpha[i])
        if L == H: print("L==H"); return 0
        
        eta = 2.0 * ds.K[i, j] - ds.K[i, i] - ds.K[j, j]
        if eta >= 0: print("eta >= 0"); return 0

        ds.alpha[j] -= ds.labels[j] * (err_i - err_j) / eta
        ds.alpha[j] = clip_alpha(ds.alpha[j], L, H)
        ds.err[j] = [1, err_j]

        if abs(ds.alpha[j] - alpha_j_pre) < 1e-5:
            print("alpha j is not moving enough")
            return 0

        ds.alpha[i] += ds.labels[i] * ds.labels[j] * (alpha_j_pre - ds.alpha[j])
        ds.err[i] = [1, err_i]

        b1 = ds.b - err_i - ds.labels[i] * (ds.alpha[i] - alpha_i_pre) * ds.K[i, i] \
                          - ds.labels[j] * (ds.alpha[j] - alpha_j_pre) * ds.K[j, i]
        b2 = ds.b - err_j - ds.labels[i] * (ds.alpha[i] - alpha_i_pre) * ds.K[i, j] \
                          - ds.labels[j] * (ds.alpha[j] - alpha_j_pre) * ds.K[j, j]
        
        if 0 < ds.alpha[i] and ds.alpha[i] < ds.C:
            ds.b = b1
        elif 0 < ds.alpha[j] and ds.alpha[j] < ds.C:
            ds.b = b2
        else:
            ds.b = (b1 + b2) / 2
        
        return 1
    else:
        return 0


if __name__=='__main__':
    dat, lbl = load_dataset('testSetRBF.txt')
    alpha3, b3 = smo_kernel(dat, lbl, 200, 0.0001, ('rbf', 1.3), 10000)

    sv = dat[alpha3 > 0]   # support vectors
    sv_label = lbl[alpha3 > 0]
    m = len(alpha3)
    # k = np.zeros(m)

    # training error
    err_cnt = 0
    for ind in range(m):
        ker_i = calculate_kernel(sv, dat[ind, :], ('rbf', 1.3))
        pred = (alpha3[alpha3 > 0]*sv_label).dot(ker_i) + b3
        
        if pred * lbl[ind] < 0:
            err_cnt += 1   
    print("Training error rate is {}".format(err_cnt / m))

    # test error
    dat_test, label_test = load_dataset('testSetRBF2.txt')
    m_test = dat_test.shape[0]
    err_cnt = 0
    for ind in range(m_test):
        ker_i = calculate_kernel(sv, dat_test[ind, :], ('rbf', 1.3))
        pred = (alpha3[alpha3 > 0]*sv_label).dot(ker_i) + b3
        
        if pred * label_test[ind] < 0:
            err_cnt += 1   
    print("Training error rate is {}".format(err_cnt / m_test))
