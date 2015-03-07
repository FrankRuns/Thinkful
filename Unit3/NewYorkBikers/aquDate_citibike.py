import requests
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas.io.json import json_normalize

r = requests.get('http://www.citibikenyc.com/stations/json')

key_list = []
for station in r.json()['stationBeanList']:
	for k in station.keys():
		if k not in key_list:
			key_list.append(key_list)

df = json_normalize(r.json()['stationBeanList'])

df['availableBikes'].hist()
plt.show()

df['totalDocks'].hist()
plt.show()

# the .desribe operator was useful for this

# What is the mean number of bikes in a dock? What is the median? How does this change if we remove the stations that aren't in service?

df.availableBikes.mean()
df.availableBikes.median()

df1 = df[df.statusValue == 'In Service']

df1.availableBikes.mean()
df1.availableBikes.median()

# What is the mean number of bikes in a dock? What is the median? How does this change if we remove the stations that aren't in service?

# so the mean goes from 8.30 to 8.35... barely changes but increases slightly which makes sense. same # of available bikes over slightly fewer stations
# median does not change 6 = 6. Also makes sense since most frequent # of bikes available is between 4.5 and 8