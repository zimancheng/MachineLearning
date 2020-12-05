from  regression import *

import random
from bs4 import BeautifulSoup
import numpy as np

def scrape_page(X, Y, file_path, year, pieces, price):
    """Scrape the html file and create X matrix and Y vector.

    Args:
        X: X matrix
        Y: Y vector
        file_path: path to the html file
        year: the year of Lego set
        pieces: number of pieces of Lego set
        price: original price of Lego set
    """
    with open(file_path, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, features='lxml')

    i = 1
    cur_row = soup.find_all('table', r = "%d" % i)
    while len(cur_row) != 0:
        cur_row = soup.find_all('table', r = "%d" % i)
        title = cur_row[0].find_all('a')[1].text.lower()

        # find for tag brand new
        if title.find('new') > -1 or title.find('nisb') > -1:
            new_flag = 1.0
        else:
            new_flag = 0.0
        
        # find for tag sold
        sold_unicode = cur_row[0].find_all('td')[3].find_all('span')
        if len(sold_unicode) == 0:
            print('Product #%d is not in sale.' % i)
        else:
            # get current price
            sold_price = cur_row[0].find_all('td')[4]
            price_str = sold_price.text.replace('$', '').replace(',', '')

            if len(sold_price) > 1:
                price_str = price_str.replace('Free shipping', '')
            selling_price = float(price_str)

            if selling_price > price * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (year, pieces, new_flag, price, selling_price))
                X.append([year, pieces, new_flag, price])
                Y.append(selling_price)

        i += 1
        cur_row = soup.find_all('table', r = "%d" % i)

def data_collection():
    """Collect data from html files and return X matrix and Y"""
    X = []
    Y = []
    scrape_page(X, Y, './setHtml/lego8288.html', 2006, 800, 49.99)
    scrape_page(X, Y, './setHtml/lego10030.html', 2002, 3096, 269.99)
    scrape_page(X, Y, './setHtml/lego10179.html', 2007, 5195, 499.99) 
    scrape_page(X, Y, './setHtml/lego10181.html', 2007, 3428, 199.99)
    scrape_page(X, Y, './setHtml/lego10189.html', 2008, 5922, 299.99)
    scrape_page(X, Y, './setHtml/lego10196.html', 2009, 3263, 249.99)

    return np.array(X), np.array(Y)

def cross_validation(dataset, labels, num_val=10):
    """Use """

    m, n = dataset.shape
    ind_list = np.arange(m)
    err_mat = np.zeros((num_val, 30))

    for i in range(num_val):
        # Create train and test set
        np.random.shuffle(ind_list)
        train = ind_list[:int(0.9 * m)]
        test = ind_list[int(0.9 * m):]

        x_train = dataset[train]
        y_train = labels[train]
        x_test = dataset[test]
        y_test = labels[test]

        theta_mat = ridge_test(x_train, y_train)    # use 30 different lambda values to fit ridge regression

        for j in range(30):        # 30 different lambda values
            mean_train = np.mean(x_train, axis=0)
            var_train = np.std(x_train, axis=0)
            x_test = (x_test - mean_train) / var_train

            y_pred = x_test.dot(theta_mat[j]) + np.mean(y_train)    
            err_mat[i, j] = calculate_rss(y_test, y_pred)
        
    mean_err = np.mean(err_mat, axis=0)
    min_err = min(mean_err)
    best_theta = theta_mat[min_err == mean_err]

    x_mean = np.mean(dataset, axis=0)
    x_std = np.std(dataset, axis=0)
    unreg_theta = best_theta / x_std

    print('predicted selling price = {:+.4f}{:+.4f}*year{:+.4f}*pieces{:+.4f}*new_product{:+.4f}*original price'.format(\
        -x_mean.dot(unreg_theta[0]) + np.mean(labels), unreg_theta[0, 0], unreg_theta[0, 1], unreg_theta[0, 2], unreg_theta[0, 3]))


def ridge_test(dataset, labels):
    """Test for ridge regression."""
    std_dataset = standardization(dataset)
    std_labels = labels - np.mean(labels)

    num_tests = 30
    theta_mat = np.zeros((num_tests, dataset.shape[1]))

    for i in range(num_tests):
        theta_mat[i] = ridge_regression(std_dataset, std_labels, np.exp(i - 10))

    return theta_mat

if __name__ == '__main__':
    x, y = data_collection()
    cross_validation(x, y, num_val=10)




    
    