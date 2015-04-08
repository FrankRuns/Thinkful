import urllib2
import pandas as pd 

# Declare links from where to get data
links = ['http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_1996.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_1997.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_1998.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_1999.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2000.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2001.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2002.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2003.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2004.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2005.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2006.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2007.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2008.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/Stormdata_2009.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/stormdata_2010.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/stormdata_2011.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/stormdata_2012.csv',
         'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/legacy/stormdata_2013.csv']

# Download csv files to working directory
for link in links:
    response = urllib2.urlopen(link)
    csv = response.read()

    csvstr = str(csv).strip("b'")

    lines = csvstr.split("\\n")
    f = open("link_#" + str(links.index(link)+1), "w")
    for line in lines:
        f.write(line + "\n")
    f.close()

# Create list of csv file names
csvs = []
for i in range(1,(2014-1996)):
    csvs.append("link_#" + str(i))

# create large data frame comprised of all annual storm data files
data = pd.read_csv('link_#1')
for el in csvs:
    newdata = pd.read_csv(el)
    data = pd.DataFrame.append(data, newdata)

# Only take columns we want... leaves out some of the metadata stuff in the last several columns
data[data.columns[0:48]]

# Send dataframe to csv
data.to_csv('allData.csv')
