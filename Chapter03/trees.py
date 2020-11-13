from math import log

def calculate_shannon_entropy(dataset):
    """
        Function: calculate_shannon_entropy
        Calculate the Shannon Entropy of a dataset.
        :params:dataset - a list of lists, each list inside it is an example and the last element of which is the class label.
        :return:a float number, the Shannon Entropy of the input dataset
    """
    # use dict to store the class labels and occurances
    num_examples = len(dataset)
    class_cnt_dict = {}
    for example in dataset:
        class_label = example[-1]
        class_cnt_dict[class_label] = class_cnt_dict.get(class_label, 0) + 1
    # calculate the probability of each class and sum
    shannon_entropy = 0.0
    for key in class_cnt_dict:
        class_prob = class_cnt_dict[key]/num_examples
        shannon_entropy -= class_prob*log(class_prob, 2)
    return shannon_entropy

def create_sample_dataset():
    """
        Function: create_sample_dataset
        Create a dataset with required format to test other functions in this module
        :params:None
        :return:return a list of lists, a list of feature names
    """
    dataset = [[1, 1, 'yes'], 
               [1, 1, 'yes'], 
               [1, 0, 'no'], 
               [0, 1, 'no'], 
               [0, 1, 'no']]
    ftr_names = ['no surfacing', 'flippers']
    return dataset, ftr_names

def split_dataset_by_ftr_val(dataset, ftr_ind, ftr_val):
    """
        Function:split_dataset_by_ftr_val
        Renders the new dataset after splitting.
        The new dataset contains all the examples in which the given feature equals the given value.
        Note that the feature value has to be categorical, not numeric.
        :params:dataset - the dataset to be splitted
        :params:ftr_ind - the index of the feature as the split
        :params:ftr_val - the value of the feature as the split
        :return:a new dataset
    """
    # get all the examples with the feature value
    ret_dataset = []
    for example in dataset:
        if example[ftr_ind] == ftr_val:
            ret_dataset.append(example[:ftr_ind] + example[ftr_ind + 1:]) #omit the feature itself
    return ret_dataset

def choose_best_split_feature(dataset):
    """
        Function:choose_best_split_feature
        Scan over all features and their values to find the best feature to split on.
        The best is defined as having the highest information gain, which equals the most reduced entropy.
        :params:dataset - the input dataset to be calculated
        :return:the index of the best feature
    """
    # get the number of features & the base entropy number
    ftr_num = len(dataset[0]) - 1
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0.0; best_feature_ind = -1
    # get the values of each feature and their summed entropy values
    # calculate the infogain of each feature and update the best feature and the biggest infogain
    for i in range(ftr_num):
        ftr_vals = [example[i] for example in dataset]
        uniq_ftr_val = set(ftr_vals)
        ftr_entropy = 0.0
        for val in uniq_ftr_val:
            sub_dataset = split_dataset_by_ftr_val(dataset, i, val)
            prob_split =  float(len(sub_dataset)/len(dataset))
            ftr_entropy += prob_split*calculate_shannon_entropy(sub_dataset)
        info_gain = base_entropy - ftr_entropy
        if (best_info_gain < info_gain):
            best_info_gain = info_gain
            best_feature_ind = i
    return best_feature_ind

def majority_count(class_list):
    """
        Function:majority_count
        Counts the occurance of each class in the class list and return the most frequent one
        :params:class_list - a list containing different classes
        :return:the name of the most frequent class
    """
    # create a dict to count classes
    class_cnt_dict = {}
    for label in class_list:
        class_cnt_dict[label] = class_cnt_dict.get(label, 0) + 1  
    class_cnt_dict = sorted(class_cnt_dict.items(), \
                            key = lambda x: x[1], reverse=True) #sort dict on value and store all 
    return class_cnt_dict[0][0]                                 #key-value pair in a list

def build_tree(dataset, ftr_list):
    """
        Function:build_tree
        Build a classification tree based on shannon entropy in the form of a nested dict
        :params:dataset - the dataset to be splitted
        :params:ftr_list - the list of features to be considered
        :return:return the leave or the subtree
    """
    #  Stop when all classes are equal
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # Return majority vote when no more features
    if len(ftr_list) == 0:
        return majority_count(class_list)

    # Choose the best feature to build the first node and for each of its value (branch), build a subtree
    best_ftr_ind = choose_best_split_feature(dataset)
    best_ftr = ftr_list[best_ftr_ind]
    
    ftr_list.remove(best_ftr)
    ftr_vals = [example[best_ftr_ind] for example in dataset]
    uniq_ftr_val = set(ftr_vals)

    tree = {best_ftr:{}} # {best_ftr:{value1: subtree, value2: subtree, ...}}
    for val in uniq_ftr_val:
        sub_ftr_list = ftr_list[:]
        tree[best_ftr][val] = build_tree(split_dataset_by_ftr_val(dataset, best_ftr_ind, val),\
                                         sub_ftr_list)
    return tree


