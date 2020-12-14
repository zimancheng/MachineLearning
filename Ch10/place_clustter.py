import numpy as np
import matplotlib.pyplot as plt 
import kmeans
import geocoder
from time import sleep
import my_constants as API_KEY
import math


def place_finder(file_name, save_file):
    """Find the latidude and longitude of places on the input file.
    
    Args:
        file_name: the file containing all places to be found
        save_file: txt file contains original info and latitude & longitude info
    Returns:
        ndarray containing latitude and longitude info of all locations
    """
    fw = open(save_file, 'w')
    ll_coords = []

    for line in open(file_name, 'r').readlines():
        line = line.strip()
        strs = line.split('\t')
        addr = strs[1] + ', ' + strs[2]
        
        try:
            g = geocoder.bing(addr, key=API_KEY.bing_api_key)
            ll_coord = g.latlng
        except:
            print('error fetching')
        
        fw.write('{}\t{}\t{}\n'.format(line, ll_coord[0], ll_coord[1]))
        ll_coords.append(ll_coord)
        sleep(1)
    
    fw.close()    
    return np.array(ll_coords)


def distance_SLC(p1, p2):
    """Compute the distance between p1 and p2 using the spherical law of consines.
    
    Args:
        p1: array of [lat, lng] of place1
        p2: array of [lat, lng] of place1
    Return:
        float number as the distance between two locations
    """
    lat1 = p1[0] * math.pi / 180
    lat2 = p2[0] * math.pi / 180
    lng1 = p1[1] * math.pi / 180
    lng2 = p2[1] * math.pi / 180

    cos = math.cos(lat2) * math.cos(lat1) * math.cos(lng2 - lng1) + math.sin(lat1) * math.sin(lat2)
    
    return math.acos(cos) * 6371.0

def club_clusters(file_name, save_file, save_img, k=5):
    """Cluster locations using Bisecting Kmeans with LSC distance and plot them on map.
    
    Args:
        file_name: the file containing all places to be found
        save_file: txt file contains original info and latitude & longitude info
        save_img: path to save image 
        k: number of clusters. Default: 5    
    """
    ll_coords = place_finder(file_name, save_file)
    centroids, clusters = kmeans.bisect_kmeans(ll_coords, k, dist_calc=distance_SLC)

    fig = plt.figure()
    coords_limit = [0.1, 0.1, 0.8, 0.8]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    axprops = dict(xticks=[], yticks=[])

    ax0 = fig.add_axes(coords_limit, label='ax0', **axprops)
    map_img = plt.imread('Portland.png')
    ax0.imshow(map_img)

    ax1 = fig.add_axes(coords_limit, label='ax1', frameon=False)
    for i in range(k):
        ax1.scatter(ll_coords[clusters[:, 0] == i, 1], ll_coords[clusters[:, 0] == i, 0], c=colors[i])
    
    ax1.scatter(centroids[:, 1], centroids[:, 0], marker='+', s=200)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.savefig(save_img)

if __name__ == '__main__':
    club_clusters('portlandClubs.txt', 'places.txt', 'clubs_on_map_5.png', 5)

