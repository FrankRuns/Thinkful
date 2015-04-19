import pandas as pd 
import numpy as np 
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
import sklearn.metrics as skm
from scipy import stats

# Download and subset the data
data = pd.read_csv('/Users/frankCorrigan/ThinkfulData/allData.csv', low_memory=False)
md = data.ix[:,['BEGIN_DATE_TIME', 'END_DATE_TIME', 'EVENT_TYPE', 'STATE', 'MONTH_NAME', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'SOURCE']] # left out injuries and deaths because data is too incomplete (lists Katrina as no deaths)

# Start cleaning data immediatly. Replace NaNs with 0 and drop dates that are 0
md = md.fillna(0)
md = md.drop(md.index[md['BEGIN_DATE_TIME']==0])

# Calculate storm duration in minutes
helper = []
for el in md['BEGIN_DATE_TIME']:
	if len(el) < 17:
		helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
	else:
		helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

md['BEGIN_DATE_TIME_CLN'] = helper

helper = []
for el in md['END_DATE_TIME']:
	if len(el) < 17:
		helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
	else:
		helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

md['END_DATE_TIME_CLN'] = helper

md['DURATION'] = (md['END_DATE_TIME_CLN'] - md['BEGIN_DATE_TIME_CLN']) / 60

# There turns out to be some bad data in here where duration is negative (start date after end date)
md = md.drop(md.index[md['DURATION']<0])
md = md.drop(md.index[md['DURATION']>50000]) # storms lasting over 34 days

# Fix the property and crop damage data (change from string 5K to float 5000.0)
# Enter to feature as a string
def removeKMBT(ds, feature):
	ds[feature] = map(lambda x: str(x), ds[feature])
	ds[ds[feature] == 'nan'] = str(0)
	ds[feature] = map(lambda x: x.replace(".", ""), ds[feature])
	ds[feature] = map(lambda x: x.replace("K", "000"), ds[feature])
	ds[feature] = map(lambda x: x.replace("M", "000000"), ds[feature])
	ds[feature] = map(lambda x: x.replace("B", "000000000"), ds[feature])
	ds[feature] = map(lambda x: x.replace("T", "000000000000"), ds[feature])
	ds[feature] = map(lambda x: float(x), ds[feature])

removeKMBT(md, 'DAMAGE_PROPERTY')
removeKMBT(md, 'DAMAGE_CROPS')

# Problems with event type names, month names. Remove repeates, remove whitespace
md['EVENT_TYPE'] = map(lambda x: x.replace("WINTER WEATHER", "Winter Weather"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("Marine Tropical Storm", "Tropical Storm"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("High Snow", "Winter Weather"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("Hurricane (Typhoon)", "Hurricane"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("Heavy Wind", "Strong Wind"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("Excessive ", ""), md['EVENT_TYPE'])
md['MONTH_NAME'] = map(lambda x: x.replace(" ", ""), md['MONTH_NAME'])

# 1. 75% of these events have no property damage recorded and 97% have no crop damage reported
md = md[md['DAMAGE_PROPERTY']!=0]

# Predict event type based on location, month, duration, and property damage
md['EVENT_TYPE'] = pd.Categorical(md['EVENT_TYPE']).labels
md['MONTH_NAME'] = pd.Categorical(md['MONTH_NAME']).labels
md['STATE'] = pd.Categorical(md['STATE']).labels

# Split training and test data
test_idx = np.random.uniform(0, 1, len(md)) <= 0.3
train = md[test_idx==True]
test = md[test_idx==False]

# Train the model
train_target = train['EVENT_TYPE']
train_data = train.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'DAMAGE_PROPERTY']]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

print rfc.oob_score_

test_target = test['EVENT_TYPE']
test_data = test.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'DAMAGE_PROPERTY']]
test_pred = rfc.predict(test_data)

print ("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))
print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))

single_test_target = test['EVENT_TYPE'].iloc[40]
single_test_data = [44, 4, 960, 4000]
single_test_pred = rfc.predict(single_test_data)

#######################################
td = md.ix[:,['EVENT_TYPE', 'DURATION']]
grouped = td.groupby('EVENT_TYPE')
g
print 'Total damage per storm type...'
print g[0:10]

# REMOVE OTHER

# Marine Thunderstorm Wind = Thunderstorm Wind
# Hail = Marine Hail
# Excessive Heat = Heat
# Cold/Wind Chill = High Snow = Extreme Cold/Wind Chill = Extreme Snow Cold/Wind Chill
def consolEvents(data):
	data['EVENT_TYPE'] = map(lambda x: x.replace("Dust Devil", "Tornado"), data['EVENT_TYPE'])	
	data['EVENT_TYPE'] = map(lambda x: x.replace("Funnel Cloud", "Tornado"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Debris Flow", "Landslide"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Tropical Storm", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Tropical Storm", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Hurrican (Typhoon)", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Tropical Depression", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Blizzard", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Heavy Snow", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Ice Storm", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Lake-Effect Snow", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("WINTER WEATHER", "Winter Weather"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Sleet", "Winter Weather"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Heavy Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine High Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Strong Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Thunderstorm Wind", "Thunderstorm Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Hail", "Hail"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Excessive Heat", "Heat"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Cold/Wind Chill", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("High Snow", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Extreme Cold/Wind Chill", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])