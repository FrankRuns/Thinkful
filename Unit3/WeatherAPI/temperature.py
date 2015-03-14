import datetime
import requests
import sqlite3 as lite
import collections
import pandas as pd
import matplotlib.pyplot as plt 

# API queries to forecast.io look like below, TIME = Unix timestamp
# https://api.forecast.io/forecast/APIKEY/LATITUDE,LONGITUDE,TIME

cities = {"Boston": '42.3319,-71.0201',
          "Cleveland": '41.4784,-81.6794',
          "Denver": '39.7392,-104.9903',
          "NewYork": '36.171800,-86.785002',
          "Seattle": '47.6097,-122.3331'
		 }

# Declare API key WHICH YOU SHOULD DEFINITELY REMOVE BEFORE COMMITING TO GH
api_key = ''
url = 'https://api.forecast.io/forecast/' + api_key

# Since we are collecting past 30 days of data, end date is today
end_date = datetime.datetime.now()

# Connect/create database 
con = lite.connect('weather.db')
cur = con.cursor()

# Create db table to store cities/weatherdata
with con:
    cur.execute('CREATE TABLE daily_temp '
    	'(day_of_reading INT, '
    	'Boston REAL, '
    	'Cleveland REAL, '
    	'Denver REAL, '
    	'NewYork REAL, '
    	'Seattle REAL)'
    )

# Declare start date which will help loop through past 30 days
query_date = end_date - datetime.timedelta(days=30)

# Populate dates for daily_temp for past 30 days
with con:
    while query_date < end_date:
        cur.execute('INSERT INTO daily_temp(day_of_reading) VALUES (?)', (int(query_date.strftime('%s')),))
        query_date += datetime.timedelta(days=1)

# Populate city and temperatue in daily_temp by looping through cities dictionary
for k,v in cities.iteritems():
    query_date = end_date - datetime.timedelta(days=30) #set value each time through the loop of cities
    while query_date < end_date:
        print(k,v)
        try:
            r = requests.get(url +'/' + v + ',' +  query_date.strftime('%Y-%m-%dT12:00:00'))
            with con:
                cur.execute('UPDATE daily_temp SET ' + k + ' = ' + str(r.json()['daily']['data'][0]['temperatureMax']) + ' WHERE day_of_reading = ' + query_date.strftime('%s'))
            query_date += datetime.timedelta(days=1)
        except ValueError:
            continue
        
# Read daily_temp into python for analysis
df = pd.read_sql_query("SELECT * FROM daily_temp", con, index_col = "day_of_reading")

# Close connection for good practice
con.close()

# Find temp range for each city
for i in df.columns:
    lower = min(df[i])
    upper = max(df[i])
    range = upper-lower
    print(i, range)

# Find the mean termperature for each city
for i in df.columns:
    print(i, df[i].mean())

# Find variance for each city
for i in df.columns:
    vrnc = 0
    meen = df[i].mean()
    for el in df[i]:
        vrnc += (el-meen)**2
print(i, vrnc)

# Plot a line chart for each cities temps over the past 30 days to find patterns
pd.DataFrame.plot(df, kind='line')
# Denver. Shown in range, variance, and the line chart.