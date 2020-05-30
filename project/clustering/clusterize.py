# Project library
#from data_read import *

# External library
#import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer

# Algorithm:

# Importing data
#print(extract_data())
X,Y = make_blobs(n_samples=300, centers=6, cluster_std=0.60, random_state=0)

# Elbow method:
def elbow_method(X):
	model = KMeans()
	visualizer = KElbowVisualizer(model,k = (2,12))
	visualizer.fit(X)
	plt.clf()
	#visualizer.show()
	return visualizer.elbow_value_


# KMeans
def clusterize(X):
	optimal_k = elbow_method(X)
	kmeans = KMeans(n_clusters=optimal_k)
	kmeans.fit(X)
	centroids = kmeans.cluster_centers_
	plt.scatter(X[:,0], X[:,1])
	for centroid in centroids:
		plt.scatter(centroid[0],centroid[1],c='r')
	plt.show()

#elbow_method(X)
clusterize(X)


# print(elbow_method(X))
# clusterize(X)
# print(X.dtype)

