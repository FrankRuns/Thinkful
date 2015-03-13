from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot
import csv
import collections
import sqlite3 as lite

url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"

r = requests.get(url)

soup = BeautifulSoup(r.content)

# soup('table')[6]

thelist = soup.findAll('tr', attrs=('class', 'tcont'))
thelist = thelist[:93]

cunts = []
for el in thelist:
	cunts.append([el.contents[1].string,
		          el.contents[3].string,
		          el.contents[15].string,
		          el.contents[21].string])

df = pd.DataFrame(cunts)
df.columns = ['Country', 'Year', 'MYears', 'FMYears']

df['MYears'] = df['MYears'].map(lambda x: int(x))
df['FMYears'] = df['FMYears'].map(lambda x: int(x))

con = lite.connect('gdpEducation.db')
cur = con.cursor()

df.to_csv('school_years.csv', header=True, index=False)

with con:
    cur.execute('CREATE TABLE school_years (country_name, _Year, _MYears, _FYears)')

with open('school_years.csv') as inputFile:
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        to_db = [line[0], line[1], line[2], line[3]]
        with con:
            cur.execute('INSERT INTO school_years (country_name, _Year, _MYears, _FYears) VALUES (?, ?, ?, ?);', to_db)

# Exploring initial Data
# All data bewteen 1999 and 2010
# Male years pretty normally distrubted
# FMale Years more spreadout and slighty skewed left (surprising)
# Median age males = 12 and median age femaile = 13 (also surpising)

# /Users/frankCorrigan/ThinkfulData/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv

with con:
    cur.execute('CREATE TABLE gdp (country_name, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010)')

with open('/Users/frankCorrigan/ThinkfulData/ny.gdp.mktp.cd_Indicator_en_csv_v2.csv') as inputFile:
    next(inputFile)
    next(inputFile)
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        with con:
            cur.execute('INSERT INTO gdp (country_name, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) VALUES ("' + line[0] + '","' + '","'.join(line[43:55]) + '");')

df1 = pd.read_sql_query("SELECT * FROM gdp", con, index_col = "country_name")

# create new table
# select * from big country list
# join small database into big --- look up types of joins
# big on left, small on right... inner join

# easier to do dataframe to csv to database

# SELECT gdp.country_name, _2005 FROM gdp JOIN school_years ON gdp.country_name = school_years.country_name 

# INSERT INTO alldata(country, gdp, myears) SELECT gdp.country_name, _2005, _MYears FROM gdp JOIN school_years ON gdp.country_name = school_years.country_name
