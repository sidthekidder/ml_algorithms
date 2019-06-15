import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig


dataset = np.genfromtxt('./usps.train', missing_values=0, delimiter=',', dtype=float)
Y100 = dataset[:, 0] # select prediction column
X100 = dataset[:, 1:] # select all other columns

# define the matrix
A = X100
# calculate the mean of each column
M = mean(A.T, axis=1)
# center columns by subtracting column means
C = A - M
# calculate covariance matrix of centered matrix
V = cov(C.T)

# eigendecomposition of covariance matrix
eigen_values, eigen_vectors = eig(V)

# sort eigenvalues and eigenvectors
idx = eigen_values.argsort()[::-1] # sorted in descending order
eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:,idx]

# visualize the 16 largest vectors
for i in range(1, 17):
	img = eigen_vectors[:, i-1:i]
	img = np.reshape(img, (16, 16))
	plt.imshow(img, cmap='gray')
	plt.show()

cumulative_variance_ratio = []
variance_till_now = 0
k70_idx, k80_idx, k90_idx = -1, -1, -1
k70, k80, k90 = [], [], []

for idx, val in enumerate(eigen_values):
	variance_till_now += val
	percentage = variance_till_now/sum(eigen_values)
	cumulative_variance_ratio.append([idx, percentage])
	if percentage <= 0.7:
		k70.append(eigen_vectors[idx])
	if percentage <= 0.8:
		k80.append(eigen_vectors[idx])
	if percentage <= 0.9:
		k90.append(eigen_vectors[idx])

	if percentage >= 0.7 and k70_idx == -1:
		k70_idx = idx
		print("70% variance achieved at " + str(idx+1) + " components.")
	if percentage >= 0.8 and k80_idx == -1:
		k80_idx = idx
		print("80% variance achieved at " + str(idx+1) + " components.")
	if percentage >= 0.9 and k90_idx == -1:
		k90_idx = idx
		print("90% variance achieved at " + str(idx+1) + " components.")

k70 = np.asarray(k70)
k80 = np.asarray(k80)
k90 = np.asarray(k90)
cumulative_variance_ratio = np.asarray(cumulative_variance_ratio)
plt.scatter(x=cumulative_variance_ratio[:,0], y=cumulative_variance_ratio[:,1])
plt.title("title")
plt.xlabel("Number of components")
plt.ylabel("% of total variance captured")
plt.show()

