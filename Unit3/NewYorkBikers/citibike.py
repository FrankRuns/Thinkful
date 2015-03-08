import time
from dateutil.parser import parse
import collections
import sqlite3 as lite
import requests
from pandas.io.json import json_normalize
import datetime
import matplotlib.pyplot as plt

r = requests.get('http://www.citibikenyc.com/stations/json')
df = json_normalize(r.json()['stationBeanList'])

con = lite.connect('citi_bike.db')
cur = con.cursor()

# create/populate reference data table
with con:
	cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')

sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

with con:
	for station in r.json()['stationBeanList']:
		cur.execute(sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))		

# create available bikes table
station_ids = df['id'].tolist() 
station_ids = ['_' + str(x) + ' INT' for x in station_ids] 

with con:
		cur.execute("CREATE TABLE available_bikes ( execution_time INT, " +  ", ".join(station_ids) + ");")

# populate available bikes table with a hour of data
for i in range(60):
	r = requests.get('http://www.citibikenyc.com/stations/json')
	exec_time = parse(r.json()['executionTime'])

	cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))
	con.commit()

	id_bikes = collections.defaultdict(int)
	for station in r.json()['stationBeanList']:
		id_bikes[station['id']] = station['availableBikes']

	for k, v in id_bikes.iteritems():
		cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")
	con.commit()

	time.sleep(60)

con.close()

hour_change = collections.defaultdict(int)
for col in df.columns:
	station_vals = df[col].tolist()
	station_id = col[1:]
	station_change = 0
	for k, v in enumerate(station_vals):
		if k < len(station_vals) - 1:
			station_change += abs(station_vals[k] - station_vals[k+1])
		hour_change[int(station_id)] = station_change

cur.execute("SELECT id, stationname, latitude, longitude FROM citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()
print "The most active station is station %s at %s latitude: %s longitude: %s " % data
print "With " + str(hour_change[max_station]) + " bikes coming and going in the hour between " + datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%H:%S') + " and " + datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%H:%S')
#print "With " + str(hour_change[max_station]) + " bikes coming and going in the hour between " + datetime.datetime.fromtimestamp(int(df.index[0])).strftime('%Y-%m-%dT%H:%M:%S') + " and " + datetime.datetime.fromtimestamp(int(df.index[-1])).strftime('%Y-%m-%dT%H:%M:%S')

plt.bar(hour_change.keys(), hour_change.values())
plt.show()