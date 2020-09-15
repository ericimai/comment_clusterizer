# Project library

# External library
import pandas as pd
import re

def import_data():
	reviews = pd.read_excel('data_EDC.xlsm',sheet_name='Result') #, index_col=0
	# columns Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])
	dados = reviews[reviews['Content'].notna()]
	# print(dados)

	return dados

def new_data():
	dados = import_data()
	new_dados= pd.DataFrame(columns=['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])

	for index, row in dados.iterrows():
	# columns Index(['Review ID', 'Location Name', 'Group Name', 'Rating', 'Content', 'Data','Source'])
		regex = r"(\.+ | e | mas | porém | porem 	)"
		res = re.sub(regex, "<space>", row['Content'], 0, re.MULTILINE)
		res1 = list(re.split("<space>| <space>|<space> ", res))
		res1 = list(filter(None, res1))
		vazio = ' '
		vazio1=''
		if vazio in res1:
			res1 = [x.strip(' ') for x in res1]
			if vazio1 in res1:
				res1 = [x.strip('') for x in res1]

		if len(res1)>1:
			for i in range(len(res1)):
				insert_r = {
					'Review ID':str(row['Review ID'])+"-"+str(i),
					'Location Name':row['Location Name'],
					'Group Name':row['Group Name'],
					'Rating':row['Rating'],
					'Content':res1[i],
					'Data':row['Data'],
					'Source':row['Source']
				}
				new_dados = new_dados.append(insert_r, ignore_index=True)
		else:
			insert_r = {
				'Review ID': str(row['Review ID']),
				'Location Name': row['Location Name'],
				'Group Name': row['Group Name'],
				'Rating': row['Rating'],
				'Content': row['Content'],
				'Data': row['Data'],
				'Source': row['Source']
			}
			new_dados = new_dados.append(insert_r,ignore_index=True)

	# print(new_dados)
	new_dados.set_index('Review ID',inplace=True)
	new_dados.to_excel('new_dados.xlsx')
	return new_dados

# df = new_data()
# print(type(df.loc['9140923-4','Content']))

