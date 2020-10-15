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

# Main();
print('Parameters: \n')
# data = import_data.import_data_bar() # dataset do bar sem quebra
# data = import_data.new_data(import_data.import_data_bar()) # dataset do bar com quebra

# data = import_data.import_data_edc() # dataset do EDC sem quebra
data = import_data.new_data(import_data.import_data_edc()) # dataset do EDC com quebra


t2, base = word_ebbending.similarity_matrix(data)
cluster_pool, cluster_index = clusterize.clusterize_share(t2, base.index.values)

# mudar aqui os nomes dos inputs e outputs
excel_input = 'Clusters_Similarity_EDC_semQuebra_main'+'.xlsx'
excel_output = 'LDA_Cluster_Similarity_EDC_semQuebra'+'.xlsx'
n_topic = 1

writer =pd.ExcelWriter(excel_input, engine = 'xlsxwriter')
for i in range(len(cluster_index)):
	# print(len(cluster_index[i]))
	# print(cluster_index[i])
	data.loc[cluster_index[i], 'Content'].to_excel(writer, sheet_name='Cluster_'+str(i))
writer.save()
print(data, '\n')
print('cluster pool len:' ,len(cluster_pool))
output_analysis.bubble_chart_v2(cluster_pool)


LDA.lda_from_clusterized_excel(n_topic,excel_input, excel_output)