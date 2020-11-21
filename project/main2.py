# Project library import
# import project.import_data as import_data
# import project.clusterize as clusterize
# import project.word_ebbending as word_ebbending
# import project.output_analysis as output_analysis
# import project.LDA as LDA
import import_data
import clusterize
import word_ebbending
import output_analysis
import LDA

# External library import
import pandas as pd
import numpy as np
		
# Main();
print('Parameters: \n')
# data = import_data.import_data_bar() # dataset do bar sem quebra
data = import_data.new_data(import_data.import_data_bar()) # dataset do bar com quebra

# data = import_data.import_data_edc() # dataset do EDC sem quebra
# data = import_data.new_data(import_data.import_data_edc()) # dataset do EDC com quebra
data_v2 = word_ebbending.get_comment_vector(data)
rating_limits = clusterize.rating_range(data_v2)
data_segment_1 = data[(data['Rating'] >= rating_limits[0]) & (data['Rating'] < rating_limits[1])]
data_segment_2 = data[(data['Rating'] >= rating_limits[1]) & (data['Rating'] < rating_limits[2])]
data_segment_3 = data[(data['Rating'] >= rating_limits[2])]
data_v2_pool = [data_segment_1, data_segment_2, data_segment_3]
sim_matrix_by_rating = []
df_pool_pool = []
data_v3_pool = []
for data_segment in data_v2_pool:
	sim_matrix = word_ebbending.similarity_matrix(data_segment)
	df_pool, data_v3, clusters, rating, weight = clusterize.clusterize_share(sim_matrix, data_segment, False)
	df_pool_pool.append(df_pool)
	data_v3_pool.append(data_v3)

# mudar aqui os nomes dos inputs e outputs
excel_input = 'Clusters_Similarity_bar_janelamento_previo'+'.xlsx'
excel_output = 'LDA_Cluster_Similarity_janelamento_previo'+'.xlsx'
n_topic = 1
writer =pd.ExcelWriter(excel_input, engine = 'xlsxwriter')
# count = 0
rating_count = 0
print('L DF POOL:', len(df_pool_pool))
for cluster_segment in df_pool_pool:
	rating_count += 1
	count = 0
	for cluster in cluster_segment:
		print(count,'\n')
		print(cluster_segment,'\n')
		cluster.to_excel(writer, sheet_name='Cluster_'+str(count) + 'Rating_'+str(rating_count))
		count += 1
writer.save()

LDA.lda_from_clusterized_excel(n_topic,excel_input, excel_output)
