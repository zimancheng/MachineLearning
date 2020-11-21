import bayes
import feedparser
import numpy as np

def calculate_top_k_words(vocab_list, full_text, k):
    """Calculate the top k frequent appearring words in vacabulary from the full text
    
    Args:
        vocab_list: a list of distinct words
        full_text: a list of words that appears in the emails
        k: integer

    Return:
        A list of top k words.
    """
    word_dict = {}
    for token in vocab_list:
        word_dict[token] = full_text.count(token)
    
    words_sorted = sorted(word_dict.items(), key = lambda x : x[1], reverse=True)
    return words_sorted[:k]

def local_words(feed1, feed0):
    """Use Naive Bayes to classify feeds from different cities.
    
    Args:
        feed1: feedparser object from one city
        feed0: feedparser object from another city

    Returns:
        vocab_list: vocabulary of the feeds
        p0_vec: the PDF of the feed distribution of feed0
        p1_vec: the PDF of the feed distribution of feed1
    """
    # Create dataset and vocabulary list
    dataset = []; labels = []; full_text = []
    min_len = min(len(feed1['entries']), len(feed2['entries']))

    for i in range(min_len):
        text1 = feed1['entries'][i]['summary']
        dataset.append(text1)
        full_text.extend(text1)
        labels.append(1)

        text0 = feed0['entries'][i]['summary']
        dataset.append(text0)
        full_text.extend(text0)
        labels.append(0)
    
    vocab_list = create_vocab_list(dataset)
    top_30_words = calculate_top_k_words(vocab_list, full_text)
    # Remove most frequent words to avoid redundancy
    for word in top_30_words:
        if word[0] in vocab_list: vocab_list.remove(word[0])    

    x_train, y_train = convert_dataset_to_array(dataset, labels, vocab_list)

    # Fit NB model using training set
    test = list(np.random.randint(0, 2*min_len, size=20))   # 20 as test set
    train = [x for x in range(2*min_len) if x not in test]

    p0_vec, p1_vec, p_y = naive_bayes_fit(x_train[train], y_train[train])

    # Validate using test set
    err_cnt = 0
    for i in range(20):
        if classify_naive_bayes(x_train[test[i]], p0_vec, p1_vec, p_y)  \
            != y_train[test[i]]: err_cnt += 1
    print("The error rate is: ", float(err_cnt) / 20)

    return vocab_list, p0_vec, p1_vec

    def get_top_words(city1, city2):
        """Get the most frequently occurring words from two cities.
        
        Args: 
            city1: the news feed from city1
            city2: the news feed from city2
        """
        vocab_list, p0_vec, p1_vec = local_words(city1, city2)
        top1 = []; top2 = []

        for i in range(len(vocab_list)):
            if p0_vec[i] > -6.0: top1.append((vocab_list[i], p0_vec[i]))
            if p1_vec[i] > -6.0: top2.append((vocab_list[i], p1_vec[i]))

        sorted_top1 = sorted(top1, key = lambda x: x[1], reverse=True)
        sorted_top2 = sorted(top2, key = lambda x: x[1], reverse=True)

        print("City1**City1**City1**City1**City1**City1**City1**City1**City1**City1**")
        for word in sorted_top1:
            print(word[0])
        
        print("City2**City2**City2**City2**City2**City2**City2**City2**City2**City2**")
        for word in sorted_top2:
            print(word[0])

        