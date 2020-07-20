# Project library

# External library
import pandas as pd

def import_data():
	reviews = pd.read_excel('data1.xlsm',sheet_name='Result', index_col=0)
	# columns Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])
	dados = reviews.dropna()
	return dados
	