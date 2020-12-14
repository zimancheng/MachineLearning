import numpy as np
import matplotlib.pyplot as plt 

def load_dataset(file_path):
    """Load dataset from the file path given, convert the data into ndarray"""
    dataset = np.loadtxt(file_path, dtype=float, delimiter='\t')
    return dataset

def calc_euclidean_dist(arr1, arr2):
    """Calculate the Euclidean distance between two arrays and return a float number."""
    dist_sq = ((arr1 - arr2) ** 2).sum()
    return np.sqrt(dist_sq)

def random_centroids(dataset, k):
    """Generate k centroids randomly given a dateset.
    
    Args:   
        dataset: ndarray of inputs. Shape (m, n)
        k: number of centriods 
    Return:
        centriods: ndarray of centroids. Shape (k, n)
    """
    m, n = dataset.shape
    centroids = np.zeros((k, n))

    for j in range(n):
        min_j = min(dataset[:, j])
        range_j = max(dataset[:, j]) - min_j
        centroids[:, j] = min_j + range_j * np.random.rand(k)
    
    return centroids

def k_means(dataset, k, dist_calc=calc_euclidean_dist, generate_centroids=random_centroids):
    """Perform K means clustering for a given dataset.
    
    Args:
        dataset: ndarray of inputs. Shape (m, n)
        k: number of centriods 
        dist_calc: calculate the distance between examples. Default calculation: Euclidean distance
        generate_centroids: initialize centroids for clustring. Default: randomly generate centroids
    Returns:
        centroids: ndarray of centroids. Shape (k, n)
        clusters: clustered dataset of shape (m, 2), 1st col as closest centroid & 2nd col as sqaured dist to centroid
    """
    m, n = dataset.shape
    clusters = np.zeros((m, 2))
    centroids = generate_centroids(dataset, k)
    centroids_changed = True

    while centroids_changed:
        centroids_changed = False
        # calculate dist between data and centroids
        for i in range(m):
            min_dist = float('inf')
            min_j = -1

            for j in range(k):
                dist = dist_calc(centroids[j], dataset[i])
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            
            if clusters[i, 0] != min_j:
                centroids_changed = True
            
            clusters[i, :] = min_j, min_dist ** 2   # col2 stores the squared dist to cluster

        # calculate cluster mean and update centroids
        for i in range(k):
            centroids[i, :] = dataset[clusters[:, 0] == i, :].mean(axis=0)
            
    return centroids, clusters

def plot_clusters(dataset, clusters, k, save_path):
    """Plot the clustered dataset"""
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    
    for i in range(k):
        ax.scatter(dataset[clusters[:, 0] == i, 0], dataset[clusters[:, 0] == i, 1], c=colors[i]) 
    
    fig.savefig(save_path)

def bisect_kmeans(dataset, k, dist_calc=calc_euclidean_dist):
    """Bisecting K Means Clustering
    
    Args:
        dataset: ndarray of inputs. Shape (m, n)
        k: number of centriods 
        dist_calc: calculate the distance between examples. Default calculation: Euclidean distance
    Returns:
        centroids: ndarray of centroids. Shape (k, n)
        clusters: clustered dataset of shape (m, 2), 1st col as closest centroid & 2nd col as sqaured dist to centroid
    """
    m, n = dataset.shape
    centroids = np.zeros((k, n))
    centroids[0] = dataset.mean(axis=0)     # initialze the first centroid

    clusters = np.zeros((m, 2))     # initialize all data to one cluster
    for i in range(m):
        clusters[i, 1] = dist_calc(centroids[0], dataset[i]) ** 2
    
    num = 1     # current number of centroids
    while num < k:
        min_rss = float('inf')

        for i in range(num):
            # obtain the data points in cluster_i and split it into two new clusters
            data_i = dataset[clusters[:, 0] == i]
            centroids_split, cluster_split = k_means(data_i, 2)

            rss_split = cluster_split[:, 1].sum()       # rss of two new clusters after split
            rss_not_split = clusters[clusters[:, 0] != i, 1].sum()      # the rss of unsplitted clusters

            if rss_not_split + rss_split < min_rss:
                min_rss = rss_not_split + rss_split
                best_cent_to_split = i
                new_centroids = centroids_split.copy()
                new_clusters = cluster_split.copy()
        
        # update centroids
        centroids[i] = new_centroids[0]
        centroids[num] = new_centroids[1]

        # assign the splitted clusters to be cluster_i and the cluster_num
        # update clusters
        new_clusters[new_clusters[:, 0] == 1, 0] = num
        new_clusters[new_clusters[:, 0] == 0, 0] = best_cent_to_split
        # the new_clusters can only be updated this way!!!
        # 0 == best_cent_to_split & 1 == num, then if bcts == 1, then all data will be cluster num
        # 0 == num & 1 == best_cent_to_split, then in iteration 1 num is 1, all data will be clustet 0
        
        clusters[clusters[:, 0] == best_cent_to_split, :] = new_clusters
        
        num += 1

    return centroids, clusters


if __name__ == '__main__':
    dat1 = load_dataset('testSet.txt')
    cent1, clust1 = k_means(dat1, 4)
    plot_clusters(dat1, clust1, 4, 'kmeans_test1.png')

    dat2 = load_dataset('testSet2.txt')
    cent2, clust2 = bisect_kmeans(dat2, 3)
    plot_clusters(dat2, clust2, 3, 'bikmeans_test2.png')

