# Objective
# Write a script called "database.py" to print out the cities with the July being the warmest month. Your script must:

# Connect to the database
# Create the cities and weather tables (HINT: first pass the statement DROP TABLE IF EXISTS <table_name>; to remove the table before you execute the CREATE TABLE ... statement)
# Insert data into the two tables
# Join the data together
# Load into a pandas DataFrame
# Print out the resulting city and state in a full sentence. For example: "The cities that are warmest in July are: Las Vegas, NV, Atlanta, GA..."
# Push your code to Github and enter the link below

import sqlite3 as lite
import pandas as pd
import sys

# if windows
con = lite.connect('C:\Users\fcorrigan\ThinkfulData\getting_started.db')

month = raw_input("Please select month: ")

with con:

	cur = con.cursor()
	cur.execute("SELECT name, state, average_high FROM cities INNER JOIN weather ON name = city WHERE warm_month = '" + month + "'")

	rows = cur.fetchall()

	if len(rows) == 0:
		print "No cities where this is the hottest month. Perhaps try July..."
		sys.exit()

	cols = [desc[0] for desc in cur.description]

	df = pd.DataFrame(rows, columns = cols)

places = []
for i in range(0, len(df)):
	places.append(df['name'][i])
	places.append(df['state'][i])

newplaces = ', '.join(places)

if len(df) == 0:
	print 'No cities where that month is the warmest'
elif len(df) == 1:
	# TODO: consider putting 'and' before last city
	print "The only city that is warmest in {0} is ".format(month) + newplaces
else:
	print "The cities that are warmest in {0} are ".format(month) + newplaces 