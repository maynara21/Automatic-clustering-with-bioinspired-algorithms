import numpy as np


def davies_bouldin(X, labels, cluster_ctr,clustertrue):
    #get the cluster assignemnts
    clusters = set(labels)
    clusters = list(clusters)
    #get the number of clusters
    num_clusters = len(clusters)
    #array to hold the number of items for each cluster, indexed by cluster number
    num_items_in_clusters = [0]*clustertrue
    #get the number of items for each cluster
    for i in range(len(labels)):
        num_items_in_clusters[int(labels[i])] += 1
        
    num_items_in_clusters=filter(lambda a: a != 0, num_items_in_clusters)
    num_items_in_clusters = list(num_items_in_clusters)
    
    max_num = -9999
    for i in range(num_clusters):
        s_i = intra_cluster_dist(X, labels, clusters[i], num_items_in_clusters[i], cluster_ctr[i])
        for j in range(num_clusters):
            if(i != j):
                s_j = intra_cluster_dist(X, labels, clusters[j], num_items_in_clusters[j], cluster_ctr[j])
                m_ij = np.linalg.norm(cluster_ctr[i]-cluster_ctr[j])
                r_ij = (s_i + s_j)/m_ij
                if(r_ij > max_num):
                    max_num = r_ij
    return max_num

def intra_cluster_dist(X, labels, cluster, num_items_in_cluster, centroid):
    total_dist = 0
    #for every item in cluster j, compute the distance the the center of cluster j, take average
    for k in range(num_items_in_cluster):
        dist = np.linalg.norm(X[labels==cluster]-centroid)
        total_dist = dist + total_dist
    return total_dist/num_items_in_cluster


def normalize_to_smallest_integers(labels):
    """Normalizes a list of integers so that each number is reduced to the minimum possible integer, maintaining the order of elements.
    :param labels: the list to be normalized
    :returns: a numpy.array with the values normalized as the minimum integers between 0 and the maximum possible value.
    """

    max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
    sorted_labels = np.sort(np.unique(labels))
    unique_labels = range(max_v)
    new_c = np.zeros(len(labels), dtype=np.int32)

    for i, clust in enumerate(sorted_labels):
        new_c[labels == clust] = unique_labels[i]

    return new_c


def dunn(labels, distances):
    """
    Dunn index for cluster validation (the bigger, the better)
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, given by the distances between its
    two closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.
    
    The bigger the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = normalize_to_smallest_integers(labels)

    unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
    max_diameter = max(diameter(labels, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(labels, distances):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    """
    labels = normalize_to_smallest_integers(labels)
    n_unique_labels = len(np.unique(labels))

    min_distances = np.zeros((n_unique_labels, n_unique_labels))
    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
    return min_distances


def diameter(labels, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :returns:
    """
    labels = normalize_to_smallest_integers(labels)
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = distances[i, ii]
    return diameters