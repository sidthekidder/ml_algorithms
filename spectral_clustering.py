import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import scipy
import sklearn
from scipy import spatial

from sklearn import datasets
from sklearn.cluster import KMeans
x, y = datasets.make_circles(n_samples=1500, factor=0.5, noise=0.05)

# plt.scatter(x[:,0], x[:,1], c=y, cmap='rainbow')
# plt.show()

def spectral_clustering(x, k, gamma, title=""):
    # construct n*n similarity matrix b/w all data points 
    # each cell = e**(-gamma * (xi - xj)**2)
    A = scipy.spatial.distance.pdist(x, metric='euclidean')
    A = scipy.spatial.distance.squareform(A)
    A = np.exp(-gamma * A)

    # D = diagonal matrix where Dii = row sum of A
    D = np.zeros(np.shape(A))
    for idx, row in enumerate(A):
        D[idx][idx] = np.sum(row) # set Dii to row sum of this row

    # compute laplacian matrix L = D - A
    L = D - A

    # compute eigenvalues and eigenvectors of L
    values, vectors = eig(L)

    # sort eigenvectors in ascending order according to eigenvalues
    # values, vectors = zip(*sorted(zip(values, vectors)))
    idx = values.argsort()
    k_smallest_eigenvalues = values[idx]
    k_smallest_eigenvectors = vectors[:,idx][:, :k]

    # k_smallest_eigenvector rows are a lower-dimensional representation of training data points
    # use sklearn.cluster.KMeans to cluster the rows into k clusters
    kmeans = KMeans(n_clusters=k).fit(k_smallest_eigenvectors)

    # generate clustering output where xi belongs to Cj if Vi belongs to Cj
    # x_copy = np.asarray(x_copy)
    plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title(title)
    plt.show()


# compare spectral_clustering and kmeans clustering with k=2 and different values of gamma
# generate scatter plot of the clusters
for gamma in [0.01, 0.5, 2, 10, 15, 20, 25, 35, 50, 75, 100, 200, 250]:
    spectral_clustering(x, k=2, gamma=gamma, title="Spectral Clustering [gamma = " + str(gamma) + "]")

