from svm_kernel import *
from os import listdir

def img2Vec(filename):
    '''
    input: the filedir of each img txt file
    output: a list of length 1024
    '''
    arr = []
    fr = open(filename)
    for i in range(32):
        line = fr.readline().strip()
        arr.extend(line)
    return arr

def load_images(file_dir):
    """Load images from directory to ndarrays
    
    Args:
        file_dir: directory to txt files containing digits
    Returns:
        dataset: ndarray of shape (m, 2014) while m is the number of txt files
        labels: array of shape (m,) -1 if image is digit 9, 1 if otherwise
    """
    train_file_list = listdir(file_dir)
    m = len(train_file_list)
    dataset = np.zeros((m, 1024))   # each digit image has 1024 features
    labels = np.zeros(m)

    for i in range(m):
        file_name = train_file_list[i]
        label = int(file_name.split('_')[0])
       
        if label == 9:
            labels[i] = -1
        else:
            labels[i] = 1
 
        dataset[i] = img2Vec('{}/{}'.format(file_dir, file_name))
    
    return dataset, labels

def classify_digits():
    """To classify digits using SVM based on RBF"""
    x_train, y_train = load_images('trainingDigits')
    x_test, y_test = load_images('testDigits')

    alpha, b = smo_kernel(x_train, y_train, 200, 0.0001, ('rbf', 10), 10000)
    sv = x_train[alpha > 0]
    sv_label = y_train[alpha > 0]
    m = len(alpha)
    
    # training error
    err_cnt = 0
    for i in range(m):
        ker_i = calculate_kernel(sv, x_train[i, :], ('rbf', 10))
        pred = (alpha[alpha > 0]*sv_label).dot(ker_i) + b

        if pred * y_train[i] < 0: err_cnt += 1
    print("Training error rate is {}".format(err_cnt / m))

    # test error
    m_test = x_test.shape[0]
    err_cnt = 0
    for i in range(m_test):
        ker_i = calculate_kernel(sv, x_test[i, :], ('rbf', 10))
        pred = (alpha[alpha > 0]*sv_label).dot(ker_i) + b

        if pred * y_test[i] < 0: err_cnt += 1
    print("Training error rate is {}".format(err_cnt / m_test))

if __name__ == '__main__':
    classify_digits()


