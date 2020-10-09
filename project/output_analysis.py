
# External library import
import matplotlib.pyplot as plt 
import numpy as np 

def bubble_chart():

	# plt.close()
	x = np.random.rand(40)
	y = np.random.rand(40)
	z = np.random.rand(40)

	print(x)
	print(len(x))
	plt.scatter(x,y, s =z*1000, alpha = 0.5)
	plt.show()

def bubble_chart_v2(cluster_pool):

	# plt.close()
	y = [] 
	z = []

	for cluster_index in range(len(cluster_pool)):
		y.append(cluster_index + 1) 
		z.append(cluster_pool[cluster_index])
	x = [3,0,4,2]
	print(x)
	print(y)
	print(z)
	plt.scatter(x,y,z, alpha = 0.5)
	plt.show()

# bubble_chart()
