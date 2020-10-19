
# External library import
import matplotlib.pyplot as plt 
import numpy as np 

def bubble_chart(x,y,z):
	plt.scatter(x,y,z, alpha = 0.5)
	plt.xlabel("Rating")
	plt.ylabel("Cluster")
	plt.title("Cluster versus Rating", size = 18)
	plt.show()
