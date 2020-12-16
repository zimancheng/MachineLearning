import numpy as np

def load_sample_data():
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    return list(map(set, dataset))

def get_c1(dataset):
    """Obtain the c1 itemsets from dataset.
    
    Args:
        dataset: list of lists containing items
    Returns:
        list of frozensets as itemsets, sorted in ascending order
    """
    c1 = []
    
    for data in dataset:
        for item in data:
            if [item] not in c1:
                c1.append([item])
    
    c1.sort()
    return list(map(frozenset, c1))

def scan_dataset(dataset, cad, min_support):
    """Scan the dataset to calculate the support of each itemset in candidate 
        and select the itemsets above min_support.

    Args:
        dataset: list of lists containing items
        cad: candidate list, list of frozensets as itemsets
        min_support: minimum support value
    Returns:
        item_list: list of itemsets that are above min_support
        support_dict: dict containing each item in candidate and their support
    """
    item_cnt = {}
    for data in dataset:
        for item in cad:
            if item.issubset(data):
                item_cnt[item] = item_cnt.get(item, 0) + 1

    num_items = len(dataset)
    item_list = []
    support_dict = {}

    for item, cnt in item_cnt.items():
        support = cnt / num_items
        support_dict[item] = support

        if support >= min_support:
            item_list.append(item)
        
    return item_list, support_dict

def generate_candidate(item_list, k):
    """Generate the candidate list of itemsets with each itemset containing k items from item_list.
    
    Args:
        item_list: list of itemsets that are above min_support
        k: number of items in a itemset in candidate list
    Returns:
        cad: candidate list of frozensets as itemsets
    """
    cad = []
    num = len(item_list)
    
    for i in range(num):
        for j in range(i + 1, num):
            # use aprior principle, if a set is not frequent then it's superset is not frequent
            # k-2: only the last different numbers of two sets are picked to form the new itemset
            l1 = list(item_list[i])[:k-2]
            l2 = list(item_list[j])[:k-2]

            l1.sort()
            l2.sort()
            if l1 == l2:
                cad.append(item_list[i] | item_list[j])
    
    return cad

def apriori(dataset, min_support=0.5):
    """Use apriori principle to obtain the frequent itemsets and their supports.
    
    Args:
        dataset: list of sets containing items
        min_support: minimum support value
    Returns:
        item_list: list of itemsets that are above min_support
        sup_dict: dict containing each item in candidate and their support
    """
    c1 = get_c1(dataset)
    l_1, sup_dict = scan_dataset(dataset, c1, min_support)

    item_list = [l_1]
    i = 0
    
    while len(item_list[i]) > 1:
        c_k = generate_candidate(item_list[i], i + 2)
        l_k, sup_dict_k = scan_dataset(dataset, c_k, min_support)
    
        item_list.append(l_k)
        sup_dict.update(sup_dict_k)

        i += 1

    return item_list, sup_dict


def calculate_confidence(freq_set, conseq_list, support_dict, rule_list, min_confidence):
    """Calculate the confidence of frequent set to consequent and update the rule list.
    
    Args:
        freq_set: a set containing frequent itemsets
        conseq_list: list of frozensets as consequents
        support_dict: dict containing each item in candidate and their support
        rule_list: list containing (antecedent, consequent, confident)
        min_confidence: minimum confidence allowed
    Returns:
        pruned_list: pruned consequent list
    """
    pruned_list = []

    for item in conseq_list:
        confidence = support_dict[freq_set] / support_dict[freq_set - item]
        if confidence >= min_confidence:
            rule_list.append((freq_set - item, item, confidence))
            pruned_list.append(item)

    return pruned_list

def merge_consequents(freq_set, conseq_list, support_dict, rule_list, min_confidence):
    n = len(conseq_list)

    if len(freq_set) > n + 1:   # ensure antecedents are more than consequents
        new_conseq_list = generate_candidate(conseq_list, n + 1)
        new_conseq_list = calculate_confidence(freq_set, new_conseq_list, support_dict, rule_list, min_confidence)

        if len(new_conseq_list) > 1:    # further merge if more than 2 consequents
            merge_consequents(freq_set, conseq_list, support_dict, rule_list, min_confidence)

def generate_rules(freq_list, support_dict, min_confidence=0.7):
    """Generate associate rules for all frequent sets.
    
    Args:
        freq_list: list of fronzensets as frequent sets
        support_dict: dict containing each item in candidate and their support
        min_confidence: minimum confidence allowed
    Returns:
        rule_list: list containing (antecedent, consequent, confident)
    """
    rule_list = []

    for i in range(1, len(freq_list)):      # freq_list[0] only contains 1 item set
        for freq_set in freq_list[i]:
            
            conseq_list = [frozenset([item]) for item in freq_set]
            conseq_list = calculate_confidence(freq_set, conseq_list, support_dict, rule_list, min_confidence)

            if i > 1:
                merge_consequents(freq_set, conseq_list, support_dict, rule_list, min_confidence)

    return rule_list


