from bs4 import BeautifulSoup
import requests
import os
from urllib2 import urlopen, URLError, HTTPError
import pandas as pd
import gzip

# Get the webpage and parse the content with Beautiful Soup
baseurl = 'http://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
r = requests.get(baseurl)
soup = BeautifulSoup(r.content)

# Grab all the links you need
links = []
for a in soup.find_all('a', href=True):
    links.append(a['href'])
links = links[7:72]

# Download all the data in links array
for link in links:
    f = urlopen(baseurl+link)
    with open(os.path.basename(baseurl+link), "wb") as local_file:
        local_file.write(f.read())

# create large data frame comprised of all annual storm data files
data = pd.read_csv(gzip.open('StormEvents_details-ftp_v1.0_d1950_c20140824.csv.gz'))
iterlinks = iter(links)
next(iterlinks)
for el in iterlinks:
    newdata = pd.read_csv(gzip.open(el), low_memory=False, error_bad_lines=False)
    data = pd.DataFrame.append(data, newdata)
    print 'Added ' + el

data.to_csv('allData.csv')
