import numpy as np
import matplotlib.pyplot as plt 

def create_sample_dataset():
    """Create a sample dataset for testing naive Bayes

    Returns:
        pos_list: posts to be classified
        labels: the label of each posting in pos_list
    """
    pos_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    labels = [0, 1, 0, 1, 0, 1]   # 1 is abusive, 0 is not
    return pos_list, labels

def create_vocab_list(dataset):
    """Create the vocabulary list for a given dataset

    Args:
        dataset: nested list of strings
    Returns:
        list of vocabulary of the dataset
    """
    vocab_set = set()
    for post in dataset:
        vocab_set |= set(post)
    return list(vocab_set)

def word_2_vec_set(vocab_list, word_set):
    """Convert word set to one-hot vector. 1 stands for the word in vocab_list, otherwise 0.

    Args:
        vocab_list: the vocabulary list
        word_set: set of words in a document or email
    Retuns:
        List of integers with the length of vocabulary list
    """
    vec = [0] * len(vocab_list)
    
    for word in word_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] = 1
        else:
            print(f"The world: {word} is not in my Vocabulary!")
    
    return vec

def word_2_vec_bag(vocab_list, word_set):
    """Convert word set to a bag of words vector.

    Args:
        vocab_list: the vocabulary list
        word_set: set of words in a document or email
    Retuns:
        List of integers with the length of vocabulary list
    """
    vec = [0] * len(vocab_list)
    
    for word in word_set:
        if word in vocab_list:
            vec[vocab_list.index(word)] += 1    # increment 1 when word in vocabulary list
        else:
            print(f"The world: {word} is not in my Vocabulary!")
    
    return vec


def naive_bayes_fit(x_train, y_train):
    """Fit a naive Bayes model to the dataset.

    Args:
        x_train: Input one hot vectors. Ndarray of shape (m, n)
        y_train: Labels of input. Array of shape (m,)
    Returns:
        p0_vec: p(x|y=0), a ndarray of hape (n,)
        p1_vec: p(x|y=0) a ndarray of shape (n,)
        p1: p(y=1). floating point
    """
    m, n = x_train.shape   # to avoid product of probabilities = 0
    p1 = sum(y_train) / m

    p0_num = np.sum(x_train[y_train == 0], axis=0) + 1.0
    p1_num = np.sum(x_train[y_train == 1], axis=0) + 1.0

    p0_dem = x_train[y_train == 0].sum() + 2.0
    p1_dem = x_train[y_train == 1].sum() + 2.0

    p0_vec = log(p0_num / p0_dem)
    p1_vec = log(p1_num / p1_dem)

    return p0_vec, p1_vec, p1

def classify_naive_bayes(x_vec, p0_vec, p1_vec, p1):
    """Using naive bayes to classify if a x vector is abusive or not.

    Args: 
        x_vec: an one hot vector of a post. List of length n.
        p0_vec: array of length n.
        p1_vec: array of length n.
        p1: p(y=1). floating point

    Returns:
        1: abusive post. 0: non-abusive post.
    """
    x_arr = np.array(x_vec)
    p0_y = x_arr.dot(p0_vec) + np.log(1 - p1)   # log of product is sum of log
    p1_y = x_arr.dot(p1_vec) + np.log(p1)

    if p1_y > p0_y: 
        print("This posting is classified as abusive.")
        return 1
    else:
        print("This posting is classified as non-abusive.")
        return 0

def convert_dataset_to_array(dataset, labels, vocab_list):
    """Convert dataset in list into ndarray for train Naive Bayes Model"""
    m = len(dataset)
    n = len(vocab_list)
    x_list = []
    
    for i in range(m):
        x_list.append(word_2_vec_set(vocab_list, dataset[i]))
    
    x_train = np.array(x_list)
    y_train = np.array(labels)
    return x_train, y_train

















                