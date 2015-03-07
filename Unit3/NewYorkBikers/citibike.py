# these libraries used for retireving the data
import requests
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas.io.json import json_normalize
# these libraries used for putting the data in database and manipulating time
import sqlite3 as lite
import time
from dateutil.parser import parse 
import collections

# function to set up the database and initialize with data from first call
def initializeBikeDb():

	r = requests.get('http://www.citibikenyc.com/stations/json')
	df = json_normalize(r.json()['stationBeanList'])

	# connect to database
	con = lite.connect('citi_bike.db')
	cur = con.cursor()

	# CREATE/POPULATE REFERENCE TABLE that will store our static data points and populate that table with desired datapoints
	with con:
		cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY, totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC, postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, landMark TEXT, latitude NUMERIC, location TEXT )')

	sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude, stAddress2, longitude, postalCode, testStation, stAddress1, stationName, landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

	with con:
		for station in r.json()['stationBeanList']:
			cur.execute(sql,(station['id'],station['totalDocks'],station['city'],station['altitude'],station['stAddress2'],station['longitude'],station['postalCode'],station['testStation'],station['stAddress1'],station['stationName'],station['landMark'],station['latitude'],station['location']))

	# CREATE/POPULATE AVAILABEL BIKES TABLE populate with stations id, time of query, and available # of bikes
	station_ids = df['id'].tolist() 
	station_ids = ['_' + str(x) + ' INT' for x in station_ids] 

	with con:
		cur.execute("CREATE TABLE available_bikes ( execution_time INT, " +  ", ".join(station_ids) + ");")

	#take the string and parse it into a Python datetime object
	exec_time = parse(r.json()['executionTime'])

	# add time for each request
	with con:
		cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))

	# lastly populate data points of available bikes into that table.
	id_bikes = collections.defaultdict(int) #defaultdict to store available bikes by station

	#loop through the stations in the station list
	for station in r.json()['stationBeanList']:
		id_bikes[station['id']] = station['availableBikes']

	#iterate through the defaultdict to update the values in the database
	with con:
		for k, v in id_bikes.iteritems():
			cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")

def updateBikeDb():

	r = requests.get('http://www.citibikenyc.com/stations/json')
	df = json_normalize(r.json()['stationBeanList'])

	# connect to database
	con = lite.connect('citi_bike.db')
	cur = con.cursor()

	# CREATE/POPULATE AVAILABEL BIKES TABLE populate with stations id, time of query, and available # of bikes
	station_ids = df['id'].tolist() 
	station_ids = ['_' + str(x) + ' INT' for x in station_ids] 

	#take the string and parse it into a Python datetime object
	exec_time = parse(r.json()['executionTime'])

	# # add time for each request
	# with con:
	# 	cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)', (exec_time.strftime('%s'),))

	# lastly populate data points of available bikes into that table.
	id_bikes = collections.defaultdict(int) #defaultdict to store available bikes by station

	#loop through the stations in the station list
	for station in r.json()['stationBeanList']:
		id_bikes[station['id']] = station['availableBikes']

	#iterate through the defaultdict to update the values in the database
	with con:
		for k, v in id_bikes.iteritems():
			cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + ";")