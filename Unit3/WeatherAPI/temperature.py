import datetime
import requests
import sqlite3 as lite
import collections
import pandas as pd

# API queries to forecast.io look like below, TIME = Unix timestamp
# https://api.forecast.io/forecast/APIKEY/LATITUDE,LONGITUDE,TIME

cities = {"Boston"   : '42.3319,-71.0201',
		  "Cleveland": '41.4784,-81.6794',
		  "Denver"   : '39.7392,-104.9903',
		  "NewYork" : '40.7127,-74.0059',
		  "Seattle"  : '47.6097,-122.3331'
		 }

api_key = '9e04edd20cb69c0b3258bcab4c270640'
url = 'https://api.forecast.io/forecast/' + api_key

end_date = datetime.datetime.now()

con = lite.connect('weather.db')
cur = con.cursor()

with con:
	cur.execute('CREATE TABLE daily_temp (day_of_reading INT, Boston REAL, Cleveland REAL, Denver REAL, NewYork REAL, Seattle REAL)')

query_date = end_date - datetime.timedelta(days=30)

with con:
	while query_date < end_date:
		cur.execute('INSERT INTO daily_temp(day_of_reading) VALUES (?)', (int(query_date.strftime('%s')),))
		query_date += datetime.timedelta(days=1)

for k,v in cities.iteritems():
	query_date = end_date - datetime.timedelta(days=30)
	while query_date < end_date:
		r = requests.get(url + '/' + v + ',' + query_date.strftime('%Y-%m-%dT12:00:00'))
		with con:
			cur.execute('UPDATE daily_temp SET ' + k + ' = ' + str(r.json()['daily']['data'][0]['temperatureMax']) + ' WHERE day_of_reading = ' + query_date.strftime('%s'))
		query_date += datetime.timedelta(days=1)

df = pd.read_sql_query("SELECT * FROM daily_temp", con, index_col = "day_of_reading")

con.close()

# Find temp range for each city
for i in df.columns:
	lower = min(df[i])
	upper = max(df[i])
	range = upper - lower
	print (i, range)

# find the mean termperature for each city
for i in df.columns:
	print (i, df[i].mean())

# variance for each city?
for i in df.columns:
	v = 0
	m = df[i].mean()
	for el in df[i]:
		v += (el - m)**2
	print (i, v)

# Any patterns?
import matplotlib.pyplot as plt 
pd.DataFrame.plot(df, kind='line')
# see that NY, Boston, Cleveland move in same pattern. Denver has high variability and Seattle is high temp (comparably) with very low variability

# which cities has the largest temperature changes over the time period?
# Denver. Shown in range, variance, and the line chart.