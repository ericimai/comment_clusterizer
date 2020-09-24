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

def cos_granurality(cos_vector, set_granurality, index):
	granurality = []
	groups_values = []
	groups_index = []
	groups_cluster = []
	start = 0
	end = 0.5	
	granurality.append(start)
	cluster = -1
	while start <= 1:
		# print(groups_values)
		# print(groups_index)
		# print(groups_cluster)

		start = start + set_granurality
		granurality.append(start)
		cluster = cluster + 1

		groups_values.append(list())
		groups_index.append(list())
		groups_cluster.append(list())
		print(start)

		for i in range(len(cos_vector)):
			if(cos_vector[i] < start and cos_vector[i] > (start - set_granurality)):
				groups_values[cluster].append(cos_vector[i])
				groups_cluster[cluster].append(cluster)
				groups_index[cluster].append(index[i])

	return groups_values, groups_index, groups_cluster

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

	return(cluster_pool, cluster_index)

# Main()==========================================================================================

# comment_matrix_cos, base = word_ebbending.get_comment_vector_cos(import_data.import_data())
# groups_values, groups_index, groups_cluster = cos_granurality(comment_matrix_cos, 0.005, base.index.values)

# print('groups_values: ', groups_values)
# print(len(groups_values), '\n')	
# print('groups_index: ', groups_index)
# print(len(groups_index), '\n')	
# print('groups_cluster: ', groups_cluster)
# print(len(groups_cluster), '\n')	

# dados = import_data.import_data()
# writer =pd.ExcelWriter('Clusters_COS.xlsx', engine = 'xlsxwriter')
# dados.loc[groups_index[1],'Content'].to_excel(writer,sheet_name='Cluster_1')
# dados.loc[groups_index[2],'Content'].to_excel(writer,sheet_name='Cluster_2')
# dados.loc[groups_index[3],'Content'].to_excel(writer,sheet_name='Cluster_3')
# dados.loc[groups_index[4],'Content'].to_excel(writer,sheet_name='Cluster_4')
# dados.loc[groups_index[5],'Content'].to_excel(writer,sheet_name='Cluster_5')
# dados.loc[groups_index[6],'Content'].to_excel(writer,sheet_name='Cluster_6')
# dados.loc[groups_index[7],'Content'].to_excel(writer,sheet_name='Cluster_7')
# dados.loc[groups_index[8],'Content'].to_excel(writer,sheet_name='Cluster_8')
# dados.loc[groups_index[9],'Content'].to_excel(writer,sheet_name='Cluster_9')
# dados.loc[groups_index[10],'Content'].to_excel(writer,sheet_name='Cluster_10')
# dados.loc[groups_index[11],'Content'].to_excel(writer,sheet_name='Cluster_11')
# dados.loc[groups_index[12],'Content'].to_excel(writer,sheet_name='Cluster_12')
# writer.save()


# Main_v1()==========================================================================================

# print('Parameters: \n')
# t2, base = word_ebbending.get_comment_vector_cos_v3(import_data.new_data())
# print(t2)
# print(len(t2),'\n')
# print(t2[0])
# print(len(t2[0]))
# # print(t2)
# # # clusterize_structure(t2)
# cluster_pool, cluster_index = clusterize_share(t2, base.index.values)
# # print(cluster_index)
# print(len(cluster_index[0]))
# # print(cluster_index[0])
# print(len(cluster_index[1]))
# # print(cluster_index[1])
# print(len(cluster_index[2]))
# # print(cluster_index[2])
# print(len(cluster_index[3]))
# # print(cluster_index[3])
# # print(len(cluster_index[4]))
# # print(cluster_index[4])

# dados = import_data.import_data()
# writer =pd.ExcelWriter('Clusters_COS_new_data_v3.xlsx', engine = 'xlsxwriter')
# dados.loc[cluster_index[0],'Content'].to_excel(writer,sheet_name='Cluster_0')
# dados.loc[cluster_index[1],'Content'].to_excel(writer,sheet_name='Cluster_1')
# dados.loc[cluster_index[2],'Content'].to_excel(writer,sheet_name='Cluster_2')
# dados.loc[cluster_index[3],'Content'].to_excel(writer,sheet_name='Cluster_3')
# # dados.loc[cluster_index[4],'Content'].to_excel(writer,sheet_name='Cluster_4')
# writer.save()

# Main_V2()==========================================================================================

print('Parameters: \n')
t2, base = word_ebbending.similarity_matrix(import_data.new_data())
# print(t2,'\n')
# print(len(t2),'\n')
# print(len(t2[0]),'\n')
# clusterize_structure(t2)

cluster_pool, cluster_index = clusterize_share(t2, base.index.values)
dados = import_data.import_data()
writer =pd.ExcelWriter('Clusters_Similarity_BAR.xlsx', engine = 'xlsxwriter')
for i in range(len(cluster_index)):
	print(len(cluster_index[i]))
	print(cluster_index[i])
	dados.loc[cluster_index[i], 'Content'].to_excel(writer, sheet_name='Cluster_'+str(i))
writer.save()