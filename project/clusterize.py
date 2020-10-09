# Project library
# import project.word_ebbending as word_ebbending
# import project.import_data as import_data
import word_ebbending
import import_data

# External library
from sklearn.cluster import KMeans
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np

# Elbow method:
def elbow_method(bag_of_words):
	model = KMeans()
	visualizer = KElbowVisualizer(model,k = (2,12))
	visualizer.fit(bag_of_words)
	plt.cla()
	# plt.clf()
	plt.close()
	return visualizer.elbow_value_

def clusterize(bag_of_words):
	optimal_k = elbow_method(bag_of_words)
	kmeans = KMeans(n_clusters=optimal_k)
	# print('Parameters: \n')
	return kmeans, optimal_k

def clusterize_structure(bag_of_words):
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	df_bw["clusters"] = kmeans.fit_predict(df_bw.values)
	df_bw.groupby("clusters").aggregate("mean").plot.bar()
	plt.show()

def clusterize_share(bag_of_words, index):
	cluster_pool = []
	cluster_index = [] 
	cluster_pool_porcent = []
	count_index = 0
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	df_bw["clusters"] = predict
	for cluster_sample in range(optimal_k):
		cluster_pool.append(0)
	for cluster in range(len(cluster_pool)):
		cluster_index.append(list())
		for sample in predict:
			if sample == cluster:
				cluster_pool[cluster] = cluster_pool[cluster] + 1
				cluster_index[cluster].append(index[count_index])
			count_index = count_index + 1
		count_index = 0
	print('\nTotal Sample: ', len(predict))
	print('Cluster_pool',cluster_pool)
	
	for id in range(len(cluster_pool)):
		cluster_pool_porcent.append(cluster_pool[id]/len(predict))
		print('Cluster',id,':', '%.2f' % cluster_pool_porcent[id], '%')

	return(cluster_pool, cluster_index)
