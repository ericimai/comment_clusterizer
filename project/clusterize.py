# Project library
import project.word_ebbending as word_ebbending
import project.import_data as import_data
# import word_ebbending
# import import_data

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

	return(cluster_pool, cluster_index)

# Main()==========================================================================================

# comment_matrix_angle, base = word_ebbending.get_comment_vector_angle(import_data.import_data())
# clusterize(comment_matrix_angle)
# print('Parameters: \n')
# t2, base = word_ebbending.get_comment_vector(import_data.import_data())
# print(t2)
# # clusterize_structure(t2)
# cluster_pool, cluster_index = clusterize_share(t2, base.index.values)
# # print(cluster_index)
#
# print(len(cluster_index[0]))
# # print(cluster_index[0])
# # print(cluster_index[0])
#
#
# print(len(cluster_index[1]))
# # print(cluster_index[1])
#
#
# print(len(cluster_index[2]))
# # print(cluster_index[2])
#
#
# print(len(cluster_index[3]))
# # print(cluster_index[3])
#
#
# print(len(cluster_index[4]))
# # print(cluster_index[4])
# dados = import_data.import_data()
# writer =pd.ExcelWriter('Clusters.xlsx', engine = 'xlsxwriter')
# dados.loc[cluster_index[0],'Content'].to_excel(writer,sheet_name='Cluster_0')
# dados.loc[cluster_index[1],'Content'].to_excel(writer,sheet_name='Cluster_1')
# dados.loc[cluster_index[2],'Content'].to_excel(writer,sheet_name='Cluster_2')
# dados.loc[cluster_index[3],'Content'].to_excel(writer,sheet_name='Cluster_3')
# dados.loc[cluster_index[4],'Content'].to_excel(writer,sheet_name='Cluster_4')
# writer.save()

# mains v2 ==========================================================================
print('Parameters: \n')
t2, base = word_ebbending.get_comment_vector(import_data.new_data())
print(t2)
# clusterize_structure(t2)
cluster_pool, cluster_index = clusterize_share(t2, base.index.values)
dados = import_data.new_data()
writer =pd.ExcelWriter('Clusters_v2.xlsx', engine = 'xlsxwriter')
for i in range(len(cluster_index)):
	print(len(cluster_index[i]))
	print(cluster_index[i])
	dados.loc[cluster_index[i], 'Content'].to_excel(writer, sheet_name='Cluster_'+str(i))
writer.save()