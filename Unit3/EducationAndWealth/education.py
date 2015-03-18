from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import sqlite3 as lite
# import math
import statsmodels.api as sm

# store url for school years
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"

# get the html
r = requests.get(url)

# parse the html content with bs
soup = BeautifulSoup(r.content)

# grab all elements with class of tcont and remove the last few
thelist = soup.findAll('tr', attrs=('class', 'tcont'))
thelist = thelist[:93]

# grab the fields we want; country_name, data year, school_years, male_school_years, female_school_years
cunts = []
for el in thelist:
	cunts.append([el.contents[1].string,
		          el.contents[3].string,
                  el.contents[9].string,
		          el.contents[15].string,
		          el.contents[21].string])

# convert data to pandas dataframe and define column names
df = pd.DataFrame(cunts)
df.columns = ['Country', 'DataYear', 'TYears', 'MYears', 'FYears']

# convert school years to integers
df['TYears'] = df['MYears'].map(lambda x: int(x))
df['MYears'] = df['MYears'].map(lambda x: int(x))
df['FYears'] = df['FYears'].map(lambda x: int(x))

# create sqlite3 database
con = lite.connect('gdpEducation.db')
cur = con.cursor()

# convert school data to csv before reading to database
df.to_csv('school_years.csv', header=True, index=False)

# create table to hold school data
with con:
    cur.execute('CREATE TABLE school_years (country_name, _Year, _TYears, _MYears, _FYears)')

# populate school years table
with open('school_years.csv') as inputFile:
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        to_db = [line[0], line[1], line[2], line[3], line[4]]
        with con:
            cur.execute('INSERT INTO school_years (country_name, _Year, _TYears, _MYears, _FYears) VALUES (?, ?, ?, ?, ?);', to_db)

# locally download gdp file from http://api.worldbank.org/v2/en/indicator/ny.gdp.mktp.cd?downloadformat=csv
# i store is here /Users/frankCorrigan/ThinkfulData/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv

# create table to hold gdp data
with con:
    cur.execute('CREATE TABLE gdp (country_name text, _1999 numeric, _2000 numeric, _2001 numeric, _2002 numeric, _2003 numeric, _2004 numeric, _2005 numeric, _2006 numeric, _2007 numeric, _2008 numeric, _2009 numeric, _2010 numeric)')

# populate gdp table from local csv
with open('/Users/frankCorrigan/ThinkfulData/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv') as inputFile:
    next(inputFile)
    next(inputFile)
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        with con:
            cur.execute('INSERT INTO gdp (country_name, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) VALUES ("' + line[0] + '","' + '","'.join(line[43:55]) + '");')

# create new table to hold combined gdp and school years data
with con:
    cur.execute('CREATE TABLE alldata (country, gdp, totalyears)')

# populate all data using join statement
with con:
    cur.execute('INSERT INTO alldata (country, gdp, totalyears) SELECT gdp.country_name, _2005, _TYears FROM gdp JOIN school_years ON gdp.country_name = school_years.country_name;')

# read gdp and school years data back into a pandas dataframe
lastdf = pd.read_sql_query("SELECT * FROM alldata", con, index_col = "country")

# CHEAT: take out any row with missing data
lastdf.dropna(inplace=True) 
lastdf = lastdf[lastdf['gdp'] != '']

# convert gdp and school years to floats
gdp = lastdf['gdp'].map(lambda x: float(x))
school_years = lastdf['totalyears'].map(lambda x: int(x))

# convert gdp to log in order to scale properly
log_gdp = gdp.map(lambda x: np.log(x))

# create scatterplot of gdp vs school years
colors = np.random.rand(len(gdp))
plt.scatter(log_gdp, school_years, c=colors)

# create linear model to quantify relationship of gdp and school years. Use gdp as dependent variable
y = np.matrix(gdp).transpose()
x = np.matrix(school_years).transpose()

X = sm.add_constant(x)
model = sm.OLS(y,X)
results = model.fit()

# want to see what happens is we use log_gdp instead of gdp
y = np.matrix(log_gdp).transpose()
x = np.matrix(school_years).transpose()

X = sm.add_constant(x)
model2 = sm.OLS(y,X)
results2 = model2.fit()

