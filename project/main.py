# Project library import
import project.import_data as import_data
import project.clusterize as clusterize
import project.word_ebbending as word_ebbending
import project.output_analysis as output_analysis
import project.LDA as LDA
# import import_data
# import clusterize
# import word_ebbending
# import output_analysis
# import LDA

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
print('data_v2: \n')
print(data_v2)
sim_matrix = word_ebbending.similarity_matrix(data_v2)
df_pool, data_v3, clusters, rating, weight = clusterize.clusterize_share(sim_matrix, data_v2, True)

# mudar aqui os nomes dos inputs e outputs
excel_input = 'Clusters_Similarity_bar_quebra1'+'.xlsx'
excel_output = 'LDA_Cluster_Similarity_bar_quebra1'+'.xlsx'
n_topic = 1
writer =pd.ExcelWriter(excel_input, engine = 'xlsxwriter')
count = 0
print('L DF POOL:', len(df_pool))
for cluster_segment in df_pool:
	print(count,'\n')
	print(cluster_segment,'\n')
	cluster_segment.to_excel(writer, sheet_name='Cluster_'+str(count))
	count += 1
writer.save()

LDA.lda_from_clusterized_excel(n_topic,excel_input, excel_output)
# output_analysis.bubble_chart(rating, clusters, weight)

# -----------------------------
df_pool_total =  []
for data_segment in df_pool:
	sim_matrix_segment = word_ebbending.similarity_matrix(data_segment)
	print('similarity_matrix')
	print(sim_matrix_segment,'\n')
	print(len(sim_matrix_segment),'\n')
	print('data data_segment')
	print(data_segment,'\n')
	try:
		df_pool_segment, data, clusters, rating, weight = clusterize.clusterize_share(sim_matrix_segment, data_segment, False)
		print('In Try\n')
		print(df_pool_segment, '\n')
		df_pool_total = df_pool_total + df_pool_segment
	except:
		df_pool_segment = data_segment
		print(df_pool_segment, '\n')
		print('In Except\n')

print('DF POOL CONCAT \n')
print(df_pool_total, '\n')

excel_input = 'Clusters_Similarity_Segment_c2'+'.xlsx'
excel_output = 'LDA_Cluster_Segment_c2'+'.xlsx'
writer =pd.ExcelWriter(excel_input, engine = 'xlsxwriter')
count = 0
print('L DF POOL:', len(df_pool_total))
for cluster_segment in df_pool_total:
	print(count,'\n')
	print(cluster_segment,'\n')
	cluster_segment.to_excel(writer, sheet_name='Cluster_'+str(count))
	count += 1
writer.save()

LDA.lda_from_clusterized_excel(n_topic,excel_input, excel_output)