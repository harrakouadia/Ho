
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('happyscore_income.csv')
data.sort_values('avg_income',inplace=True)
#print(data) 
income = data['avg_income']
happy = data['happyScore']
#print(happy)
richest=data[data['avg_income']>1000]
plt.scatter(income, happy )
plt.xlabel('average income')

plt.ylabel('happiness')
plt.text( data.iloc[0]['avg_income'], data.iloc[0]['happyScore'],data.iloc[0]['country'])
plt.text( data.iloc[-1]['avg_income'], data.iloc[-1]['happyScore'],data.iloc[-1]['country'])

#for dataa in data:
#print(data.avg_income)
#print(data.happyScore)
#print(data.country)
#plt.text(5, 1777, 'TEXT')
#for k,row in data.iterrows():
#    plt.text(row['avg_income'], row['happyScore'] , row['country'] )

from sklearn.cluster import KMeans

income_happy = np.column_stack((income, happy))
#print(income_happy)
km_res = KMeans(n_clusters=3).fit(income_happy)
clusters= km_res.cluster_centers_
plt.scatter(clusters[:,0], clusters[:,1],s=100)

