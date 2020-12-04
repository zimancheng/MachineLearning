import numpy as np
import matplotlib.pyplot as plt  

def load_dataset(file_path):
    """Load txt file to ndarray."""
    with open(file_path, 'r') as fr:
        header = fr.readline().strip().split('\t')
    
    x_col = [i for i in range(len(header) - 1)]
    y_col = [-1]    # last col is labels

    dataset = np.loadtxt(file_path, delimiter='\t', usecols=x_col)
    labels = np.loadtxt(file_path, delimiter='\t', usecols=y_col)

    return dataset, labels

def linear_regression(dataset, labels):
    """Compute the parameter theta for linear regression given dataset and labels.
    
    Args:
        dataset: ndarray of inputs, x0 = 1. Shape (m, n)
        labels: array of labels. Shape (m,)
    Returns:
        theta: array of parameters. Shape (n,)
    """
    # Normal equation for linear regression solver
    X_sq = dataset.T.dot(dataset)
    if np.linalg.det(X_sq) == 0.0:
        print('This is a singular matrix, cannot be inversed.')
        return
    else:
        theta = np.linalg.inv(X_sq).dot(dataset.T).dot(labels)
        return theta

def lr_test(file_path):
    """Load dataset and test for linear regression."""
    dataset, labels = load_dataset(file_path)
    theta = linear_regression(dataset, labels)
    print('Parameters of linear regression is {}'.format(theta))
    
    plot_model(dataset, labels, theta, 'lin_reg.png')
    
    cor = np.corrcoef(labels, dataset.dot(theta))
    print('Correlation coefficients between h(x) and y are {}'.format(cor[0, -1]))

def plot_model(dataset, labels, theta, save_path):
    """Plot the linear regression model on dataset."""
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 1], labels, s=10, c='b')
    ax.plot(dataset[:, 1], dataset.dot(theta), c='red', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_path)

def lwr(dataset, labels, x, tau=1.0):
    """Train a locally weighted linear regression on dataset.
    
    Args:
        dataset: ndarray of inputs, x0 = 1. 
        labels: array of labels. 
        tau: parameter determining weights. Float number
        x: the input x to be predicted, ndarray
    Returns:
        y: predicted value for input x. array
    """
    m, n = x.shape
    y = np.zeros(m)

    for i in range(m):
        W = np.diag(np.exp(-np.sum((dataset - x[i]) ** 2, axis=1) / (2 * tau ** 2)))
        theta = np.linalg.inv(dataset.T.dot(W).dot(dataset)).dot(dataset.T).dot(W).dot(labels)
        y[i] = theta.dot(x[i])
    
    return y

def lwr_test(file_path, tau_list):
    """Load dataset and test for locally weighted linear regression."""
    dataset, labels = load_dataset(file_path)
    # tau_list = [0.003, 0.01, 0.05, 0.1, 1]     # tau near 0.001 can cause exp overflow

    for tau in tau_list:
        y_pred = lwr(dataset, labels, dataset[:], tau)    
        print('LWR with tau: {}'.format(tau))
        print('Inputs: {}, predictions: {}'.format(dataset[:5], y_pred[:5]))    # test the first 5 examples in dataset
        
        fig, ax = plt.subplots()
        ax.scatter(dataset[:, 1], labels, s=10, c='b')
        ax.scatter(dataset[:, 1], y_pred[:], s=5, c='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('lwr_{}.png'.format(tau))
        
        cor = np.corrcoef(labels, y_pred)
        print('Correlation coefficients between h(x) and y are {}'.format(cor[0, -1]))

        rss = calculate_rss(labels, y_pred)
        print('The RSS of this model is {}'.format(rss))
    
def calculate_rss(labels, y_pred):
    """Calculate the RSS of a given prediction."""
    return ((labels - y_pred) ** 2).sum()


def ridge_regression(dataset, labels, lam=0.2):
    """Fit Ridge Regression to the input dataset.
    
    Args:
        dataset: ndarray of inputs, x0 = 1. Shape (m, n) 
        labels: array of labels. Shape (m,)
        lam: regularization parameter. Float number
    Returns:
        theta: array of model parameters. Shape (n,)
    """
    # Normal equation for linear regression solver
    X_sq = dataset.T.dot(dataset) + lam * np.eye(dataset.shape[1])

    if np.linalg.det(X_sq) == 0.0:
        print('This is a singular matrix, cannot be inversed.')
        return
    else:
        theta = np.linalg.inv(X_sq).dot(dataset.T).dot(labels)
        return theta

def normalization(data):
    """Normalize a ndarray"""
    _range = np.max(data) - np.min(data)  # np.max(arr, axis=0)
    return (data - np.min(data)) / _range

def standardization(data):
    """Standardize a ndarray"""
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    return (data - mean) / sd

def ridreg_test(train_path, save_path):
    """Test for ridge regression."""
    dataset, labels = load_dataset(train_path)
#     norm_datset = normalization(dataset)
#     norm_labels = normalization(labels)
    std_dataset = standardization(dataset)
    std_labels = standardization(labels)

    num_tests = 30
    theta_mat = np.zeros((num_tests, dataset.shape[1]))

    for i in range(num_tests):
        theta_mat[i] = ridge_regression(std_dataset, std_labels, np.exp(i - 10))

    fig, ax = plt.subplots()
    ax.plot(theta_mat)
    plt.xlabel('log(lambda)')
    plt.ylabel('theta')
    plt.savefig(save_path)

    return theta_mat




if __name__ == '__main__':
    lr_test('ex0.txt')
    lwr_test('ex0.txt', [0.003, 0.01, 0.05, 0.1, 1])
    # lwr_test('abalone.txt', [0.1, 1, 10])

    ridreg_test('abalone.txt', 'ridge_regression.png')