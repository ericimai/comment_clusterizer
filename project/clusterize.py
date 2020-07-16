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
def elbow_method(X):
	model = KMeans()
	visualizer = KElbowVisualizer(model,k = (2,12))
	visualizer.fit(X)
	return visualizer.elbow_value_

def clusterize_structure(X):
	df_x = pd.DataFrame(X)
	optimal_k = elbow_method(X)
	kmeans = KMeans(n_clusters=optimal_k)
	df_x["clusters"] = kmeans.fit_predict(df_x.values)
	df_x.groupby("clusters").aggregate("mean").plot.bar()
	plt.show()

def clusterize_share(X):
	df_x = pd.DataFrame(X)
	optimal_k = elbow_method(X)
	kmeans = KMeans(n_clusters=optimal_k)
	df_x["clusters"] = kmeans.fit_predict(df_x.values)
	sample_0 = 0
	sample_1 = 0
	sample_2 = 0
	sample_3 = 0
	sample_4 = 0
	for sample in kmeans.predict(X):
		if sample == 0:
			sample_0 = sample_0 + 1
		elif sample == 1:
			sample_1 = sample_1 + 1
		elif sample == 2:
			sample_2 = sample_2 + 1		
		elif sample == 3:
			sample_3 = sample_3 + 1
		elif sample == 4:
			sample_4 = sample_4 + 1
	print('\n')
	print('Total:',len(kmeans.predict(X)),'\n')
	print('Share:\n')
	c0_porc = 100*sample_0/len(kmeans.predict(X))
	print(c0_porc ,'%')
	c1_porc = 100*sample_1/len(kmeans.predict(X))
	print(c1_porc ,'%')
	c2_porc = 100*sample_2/len(kmeans.predict(X))
	print(c2_porc ,'%')
	c3_porc = 100*sample_3/len(kmeans.predict(X))
	print(c3_porc ,'%')
	c4_porc = 100*sample_4/len(kmeans.predict(X))
	print(c4_porc ,'%\n')

	share = [c0_porc,c1_porc,c2_porc,c3_porc,c4_porc]
	x = np.arange(optimal_k)
	width = 0.35
	fig, ax, = plt.subplots()
	ax.bar(x, share, width, label = "cluster")
	ax.set_xlabel('porcent')
	ax.set_xlabel('cluster')
	ax.set_title('Share')
	ax.legend()
	plt.show()

	return(kmeans.predict(X))

clusterize_share(word_ebbending.get_comment_vector(import_data.import_data()))