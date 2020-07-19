# Project library
import word_ebbending
import import_data

# External library
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np

# Elbow method:
def elbow_method(bag_of_words):
	model = KMeans()
	visualizer = KElbowVisualizer(model,k = (2,12))
	visualizer.fit(bag_of_words)
	return visualizer.elbow_value_

def clusterize(bag_of_words):
	optimal_k = elbow_method(bag_of_words)
	kmeans = KMeans(n_clusters=optimal_k)
	print('Parameters: \n')
	return kmeans, optimal_k

def clusterize_structure(bag_of_words):
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	df_bw["clusters"] = kmeans.fit_predict(df_bw.values)
	df_bw.groupby("clusters").aggregate("mean").plot.bar()
	plt.show()

def clusterize_share(bag_of_words):
	cluster_pool = []
	df_bw = pd.DataFrame(bag_of_words)
	kmeans, optimal_k = clusterize(bag_of_words)
	predict = kmeans.fit_predict(df_bw.values)
	df_bw["clusters"] = predict
	for cluster_sample in range(optimal_k):
		cluster_pool.append(0)
	for cluster in range(len(cluster_pool)):
		for sample in predict:
			if sample == cluster:
				cluster_pool[cluster] = cluster_pool[cluster] + 1
	print('\nTotal Sample: ', len(predict))
	print('Cluster_pool',cluster_pool)
	
	for id in range(len(cluster_pool)):
		cluster_pool[id] = cluster_pool[id]/len(predict)
		print('Cluster',id,':', '%.2f' % cluster_pool[id], '%')
		
	# share = [c0_porc,c1_porc,c2_porc,c3_porc,c4_porc]
	# x = np.arange(optimal_k)
	# width = 0.35
	# fig, ax, = plt.subplots()
	# ax.bar(x, share, width, label = "cluster")
	# ax.set_xlabel('porcent')
	# ax.set_xlabel('cluster')
	# ax.set_title('Share')
	# ax.legend()
	# plt.show()

	return(kmeans.predict(X))

clusterize(word_ebbending.get_comment_vector(import_data.import_data()))
# clusterize_structure(word_ebbending.get_comment_vector(import_data.import_data()))
# clusterize_share(word_ebbending.get_comment_vector(import_data.import_data()))