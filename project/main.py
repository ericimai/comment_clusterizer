# Project library import
# import project.import_data as import_data
# import project.clusterize as cluster
# import project.word_ebbending as word_ebbending
import import_data
import clusterize
import word_ebbending
import output_analysis

# External library import
import pandas as pd

# Main();
print('Parameters: \n')
data = import_data.new_data()
t2, base = word_ebbending.similarity_matrix(data)
cluster_pool, cluster_index = clusterize.clusterize_share(t2, base.index.values)

writer =pd.ExcelWriter('Clusters_Similarity_Bar_comQuebra_main.xlsx', engine = 'xlsxwriter')
for i in range(len(cluster_index)):
	# print(len(cluster_index[i]))
	# print(cluster_index[i])
	data.loc[cluster_index[i], 'Content'].to_excel(writer, sheet_name='Cluster_'+str(i))
writer.save()
print(data, '\n')
print('cluster pool len:' ,len(cluster_pool))
output_analysis.bubble_chart_v2(cluster_pool)
