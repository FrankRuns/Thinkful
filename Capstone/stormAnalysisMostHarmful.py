import pandas as pd 
import numpy as np 
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
from scipy import stats

def main():

	data = pd.read_csv('/Users/frankCorrigan/ThinkfulData/allData.csv', low_memory=False)

	# Subset smaller dataset to work with
	hd = data.ix[:,['BEGIN_DATE_TIME', 'END_DATE_TIME', 'MONTH_NAME', 'STATE', 'EPISODE_ID', 'EVENT_TYPE', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'SOURCE']]
	
	# Replace Nan's with zero's and event with no start data
	hd = hd.fillna(0)
	hd = hd.drop(hd.index[hd['BEGIN_DATE_TIME']==0])

	# Calculate each storm duration in minutes
	helper = []
	for el in hd['BEGIN_DATE_TIME']:
		if len(el) < 17:
			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
		else:
			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

	hd['BEGIN_DATE_TIME_CLN'] = helper

	helper = []
	for el in hd['END_DATE_TIME']:
		if len(el) < 17:
			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
		else:
			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

	hd['END_DATE_TIME_CLN'] = helper

	hd['DURATION'] = (hd['END_DATE_TIME_CLN'] - hd['BEGIN_DATE_TIME_CLN']) / 60
	
	# There turns out to be some bad data in here where duration is negative (start date after end date)
	hd = hd.drop(hd.index[hd['DURATION']<0])
	hd = hd.drop(hd.index[hd['DURATION']>50000])

	# Fix financial variables. Convert to string, drop K's, M's, B's, and T's, convert to float
	hd['DAMAGE_PROPERTY'] = map(lambda x: str(x), hd['DAMAGE_PROPERTY'])
	hd['DAMAGE_CROPS'] = map(lambda x: str(x), hd['DAMAGE_CROPS'])
	hd['DAMAGE_PROPERTY'] = map(lambda x: removeKMBT(x), hd['DAMAGE_PROPERTY'])
	hd['DAMAGE_CROPS'] = map(lambda x: removeKMBT(x), hd['DAMAGE_CROPS'])
	hd['DAMAGE_PROPERTY'] = map(lambda x: float(x), hd['DAMAGE_PROPERTY'])
	hd['DAMAGE_CROPS'] = map(lambda x: float(x), hd['DAMAGE_CROPS'])

	# Combine both direct and indirect deaths into deaths and convert to float
	hd['DEATHS_DIRECT'] = map(lambda x: float(x), hd['DEATHS_DIRECT'])
	hd['DEATHS_INDIRECT'] = map(lambda x: float(x), hd['DEATHS_INDIRECT'])
	hd['DEATHS'] = hd['DEATHS_DIRECT'] + hd['DEATHS_INDIRECT']
	hd['DEATHS'] = map(lambda x: float(x), hd['DEATHS'])

	# Consolidate event types based upon similar duration
	hd = hd[hd['EVENT_TYPE'] != 0]
	consolEvents(hd)

	# Calculate total damage: (Property damage + crop damage) * (1 + # of deaths)
	hd['FIN_DAMAGE'] = hd['DAMAGE_PROPERTY'] + hd['DAMAGE_CROPS']
	hd['TOTAL_DAMAGE'] = (hd['DAMAGE_PROPERTY'] + hd['DAMAGE_CROPS']) * (1 + hd['DEATHS'])

	# Initial most harmful storm
	mx = hd['TOTAL_DAMAGE'].max()
	ob = hd[hd['TOTAL_DAMAGE']==mx]
	print 'Initial Most Harmful Storm Type: ' + str(ob['EVENT_TYPE'])
	print 'Initial Most Harmful Storm Type Fin damage: ' + str(ob['FIN_DAMAGE'])
	print 'Initial Most Harmful Storm Type Deaths: ' + str(ob['DEATHS'])

	# Subset storms with reported damage. Others we assume inconsequential to this analysis
	hd = hd[hd['TOTAL_DAMAGE'] > 0]

	# Each storm (or observation) in the set is a component of a storm
	# For instance, tornado the rips through Wichita KS and then Auburn KS is 2 observations
	# Aggregate total damage by storm type
	td = hd.ix[:,['EVENT_TYPE', 'TOTAL_DAMAGE']]
	grouped = td.groupby('EVENT_TYPE')
	g = grouped.aggregate(np.sum).sort('TOTAL_DAMAGE', ascending=False)
	print 'Total damage per storm type...'
	print g[0:10]

	cownts = td['EVENT_TYPE'].value_counts()
	both = pd.concat([cownts, g], axis=1)
	both['percap'] = both['TOTAL_DAMAGE'] / both[0]
	percap = both.sort(columns='percap', ascending=False)[0:10]
	print 'Total damage per storm...'
	print percap

	# Do the same thing as above, but only for deaths
	td = hd.ix[:,['EVENT_TYPE', 'DEATHS']]
	grouped = td.groupby('EVENT_TYPE')
	g = grouped.aggregate(np.sum).sort('DEATHS', ascending=False)
	print 'Total deaths per storm type...'
	print g[0:10]

	cownts = td['EVENT_TYPE'].value_counts()
	both = pd.concat([cownts, g], axis=1)
	both['percap'] = both['DEATHS'] / both[0]
	percap = both.sort(columns='percap', ascending=False)[0:10]
	print 'Total deaths per storm...'
	print percap

	# Last, since storms are broken down by location I want to know about total storm damage
	sd = hd.ix[:,['EPISODE_ID', 'EVENT_TYPE', 'TOTAL_DAMAGE']]
	grouped = sd.groupby('EPISODE_ID')
	g = grouped.aggregate(np.sum).sort('TOTAL_DAMAGE', ascending=False)
	print 'Property damage by episode...'
	print g.head(10)

	temp = data[data['EPISODE_ID'] == int(g.index[0])].iloc[0] 
	print "Storm #1: " + str(temp['STATE']) + " " + str(temp['YEAR']) + " " +  str(temp['EVENT_TYPE']) + " " + str(temp['DEATHS_DIRECT'])
	temp = data[data['EPISODE_ID'] == int(g.index[1])].iloc[0]
	print "Storm #2: " + str(temp['STATE']) + " " + str(temp['YEAR']) + " " +  str(temp['EVENT_TYPE']) + " " + str(temp['DEATHS_DIRECT'])
	temp = data[data['EPISODE_ID'] == int(g.index[2])].iloc[0]
	print "Storm #3: " + str(temp['STATE']) + " " + str(temp['YEAR']) + " " +  str(temp['EVENT_TYPE']) + " " + str(temp['DEATHS_DIRECT']) 

	# Make some predictions
	hd['EVENT_TYPE'] = pd.Categorical(hd['EVENT_TYPE']).labels
	hd['MONTH_NAME'] = pd.Categorical(hd['MONTH_NAME']).labels
	hd['STATE'] = pd.Categorical(hd['STATE']).labels

	# Split training and test data
	test_idx = np.random.uniform(0, 1, len(hd)) <= 0.3
	train = hd[test_idx==True]
	test = hd[test_idx==False]

	# Train the model
	train_target = train['EVENT_TYPE']
	train_data = train.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'TOTAL_DAMAGE']]
	rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
	rfc.fit(train_data, train_target)

	print rfc.oob_score_

	test_target = test['EVENT_TYPE']
	test_data = test.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'TOTAL_DAMAGE']]
	test_pred = rfc.predict(test_data)

	plt.scatter(test_target, test_pred)
	plt.show()

	print ("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))
	print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))

	single_test_target = test['EVENT_TYPE'].iloc[40]
	single_test_target2 = test['EVENT_TYPE'].iloc[0]
	single_test_data = [44, 4, 960, 4000]
	single_test_data2 = [26, 9, 2040, 5000]
	single_test_pred = rfc.predict(single_test_data)
	single_test_pred2 = rfc.predict(single_test_data2)	
	single_test_pred_prob = rfc.predict_proba(single_test_data)

# Function to clean financial variables so they can be added and multiplied
# Need to convert 4K to $4,000, 3.56M to 3,560,000, etc.
def removeKMBT(num):
	if "K" in num: 
		if "." in num:
			if num[0] == ".":
				num = num[1] + "00"
			else:
				num = num.replace(".", "")
				num = num[:2]
				num = num + "00"		
		else:
			num = num.replace("K", "000")
	elif "M" in num: 
		if "." in num:
			if num[0] == ".":
				num = num[1] + "000"
			else:
				num = num.replace(".", "")
				num = num[:2]
				num = num + "00000"		
		else:
			num = num.replace("M", "000000")
	elif "B" in num: 
		if "." in num:
			if num[0] == ".":
				num = num[1] + "000000"
			else:
				num = num.replace(".", "")
				num = num[:2]
				num = num + "00000000"		
		else:
			num = num.replace("B", "000000000")
	elif "T" in num: 
		if "." in num:
			num = num.replace(".", "")
			num = num[:2]
			num = num + "00000000000"		
		else:
			num = num.replace("T", "000000000000")
	return num

# Function to consolidate storm types. For instance, sleet = winter weather
def consolEvents(data):
	data['EVENT_TYPE'] = map(lambda x: x.replace("Dust Devil", "Tornado"), data['EVENT_TYPE'])	
	data['EVENT_TYPE'] = map(lambda x: x.replace("Funnel Cloud", "Tornado"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Debris Flow", "Landslide"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Storm Surge/Tide", "Hurricane"), data['EVENT_TYPE'])	
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Tropical Storm", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Tropical Storm", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Hurricane (Typhoon)", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Tropical Depression", "Hurricane"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Blizzard", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Heavy Snow", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Ice Storm", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Lake-Effect Snow", "Winter Storm"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("WINTER WEATHER", "Winter Weather"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Sleet", "Winter Weather"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Thunderstorm Wind", "High Wind"), data['EVENT_TYPE'])	
	data['EVENT_TYPE'] = map(lambda x: x.replace("Heavy Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Strong Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine High Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Strong Wind", "High Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Thunderstorm Wind", "Thunderstorm Wind"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Marine Hail", "Hail"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Excessive Heat", "Heat"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Cold/Wind Chill", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("High Snow", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])
	data['EVENT_TYPE'] = map(lambda x: x.replace("Extreme Cold/Wind Chill", "Extreme Snow Cold/Wind Chill"), data['EVENT_TYPE'])	

# # Calculate storm duration in minutes
# def calcDuration():
# 	helper = []
# 	for el in hd['BEGIN_DATE_TIME']:
# 		if len(el) < 17:
# 			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
# 		else:
# 			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

# 	hd['BEGIN_DATE_TIME_CLN'] = helper

# 	helper = []
# 	for el in hd['END_DATE_TIME']:
# 		if len(el) < 17:
# 			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M").timetuple()))
# 		else:
# 			helper.append(time.mktime(datetime.strptime(el, "%m/%d/%Y %H:%M:%S").timetuple()))

# 	hd['END_DATE_TIME_CLN'] = helper

# 	hd['DURATION'] = (hd['END_DATE_TIME_CLN'] - hd['BEGIN_DATE_TIME_CLN']) / 60

# # Predict event type based on location, month, duration, and property damage
# def predStormType():
# 	hd['EVENT_TYPE'] = pd.Categorical(hd['EVENT_TYPE']).labels
# 	hd['MONTH_NAME'] = pd.Categorical(hd['MONTH_NAME']).labels
# 	hd['STATE'] = pd.Categorical(hd['STATE']).labels

# 	# Split training and test data
# 	test_idx = np.random.uniform(0, 1, len(hd)) <= 0.3
# 	train = hd[test_idx==True]
# 	test = hd[test_idx==False]

# 	# Train the model
# 	train_target = train['EVENT_TYPE']
# 	train_data = train.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'DAMAGE_PROPERTY']]
# 	rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
# 	rfc.fit(train_data, train_target)

# 	print rfc.oob_score_

# 	test_target = test['EVENT_TYPE']
# 	test_data = test.ix[:,['STATE', 'MONTH_NAME', 'DURATION', 'DAMAGE_PROPERTY']]
# 	test_pred = rfc.predict(test_data)

# 	print ("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))
# 	print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))

# 	single_test_target = test['EVENT_TYPE'].iloc[40]
# 	single_test_data = [44, 4, 960, 4000]
# 	single_test_pred = rfc.predict(single_test_data)

if __name__ == '__main__':
    main()