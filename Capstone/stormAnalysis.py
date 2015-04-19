import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import sklearn.metrics as skm

data = pd.read_csv('/Users/frankCorrigan/ThinkfulData/allData.csv', low_memory=False)

# Goals
# 1. Determine the most harmful storm type
# 2. Predict storm type based on known metrics
# 3. Predict damage of storm based on data that's known before storm (for instance we don't know death count before the storm)

# Hypotheses
# Most harmful storm type, based on financial damage, deaths, and injuries will be tornados
# For the classification problem, it should be relatively easy to predict storm type based on region, damage, and time of year
# For the regression problem, while I think time of year and storm type will be strongest predictors it will be interesitng to see how it varies with location

# First, we need to define 'most harmful'
# The 'damage' variables are INJURIES_DIRECT, INJURIES_INDIRECT, DEATHS_DIRECT, DEATHS_INDIRECT, DAMAGE_PROPOERTY, and DAMAGE_CROPS
# Damage = (total propoerty damage + total crop damage) * (1 + direct deaths + indirect deaths) # not sure if this works... but lets use it for now
# Need to clean up damage variables first

# Test for bad data
# for i in range(len(data['TOTAL_DAMAGE'])):
#     print i, float(data['TOTAL_DAMAGE'][i])

# Subset smaller dataset to work with
harmfulData = mini_data = data.ix[:,['EPISODE_ID', 'EVENT_TYPE', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']]
harmfulData.dropna(inplace=True)

# Fix the property and crop damage data
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

removeKMBT(harmfulData, 'DAMAGE_PROPERTY')
removeKMBT(harmfulData, 'DAMAGE_CROPS')

# Consolidate direct and indirect deaths into total deaths
harmfulData['DEATHS_DIRECT'] = map(lambda x: float(x), harmfulData['DEATHS_DIRECT'])
harmfulData['DEATHS_INDIRECT'] = map(lambda x: float(x), harmfulData['DEATHS_INDIRECT'])
harmfulData['DEATHS'] = data['DEATHS_DIRECT'] + harmfulData['DEATHS_INDIRECT']
harmfulData['DEATHS'] = map(lambda x: float(x), harmfulData['DEATHS'])

# Calculate total damage
harmfulData['DAMAGE_CROPS'][harmfulData['DAMAGE_CROPS'] > 1000000000000] = float(0) # seem to be unrealistic figures
harmfulData['TOTAL_DAMAGE'] = (harmfulData['DAMAGE_PROPERTY'] + harmfulData['DAMAGE_CROPS'])*(1+harmfulData['DEATHS'])

# Aggregate data by storm type
td = harmfulData.ix[:,['EVENT_TYPE', 'TOTAL_DAMAGE']]
grouped = td.groupby('EVENT_TYPE')
g = grouped.aggregate(np.sum).sort('TOTAL_DAMAGE', ascending=False)
print g[0:10] # most harmful storm is tornado... or is it?

cownts = td['EVENT_TYPE'].value_counts()
both = pd.concat([cownts, g], axis=1)
both['percap'] = both['TOTAL_DAMAGE'] / both[0]
both.sort(columns='percap', ascending=False)[0:10] # interesting, tsunami tops this list with tornados 2nd. Less frequent.. but more destructive

# Now, I'm curious to see where Tsunami's are occuring in the US...
data[data['EVENT_TYPE'] == 'Tsunami'].STATE # America Samoa, California, Hawaii, and Oregon

##### PREDICTING STORM TYPE

# Predict storm type. Clean data, then use random forest and initial predictors of propoert damage, crop dmaage, deaths, injuries, state, and month to predict event type
# Begin by subsetting the dataframe

md = data.ix[:,['END_DAY', 'BEGIN_DAY', 'EVENT_TYPE', 'STATE_FIPS', 'MONTH_NAME', 'STATE', 'INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'SOURCE']]
md.dropna(inplace=True)
removeKMBT(md, 'DAMAGE_PROPERTY')
removeKMBT(md, 'DAMAGE_CROPS')
md['DURATION'] = md['END_DAY'] - md['BEGIN_DAY']
md = md[md['DURATION'] > 0] # flawed. need to use duration in muntes, not days. 
md = md[md['DAMAGE_PROPERTY'] > 0]
md = md[md['EVENT_TYPE'] == 'Tornado']
# should also add tornado width and length to analysis for prediction

# Simple plot of property damage shows we will have scaling issues
plt.plot(md['DAMAGE_PROPERTY'])
md['DAMAGE_PROPERTY_SCALED'] = preprocessing.scale(md['DAMAGE_PROPERTY'])

# Quick value count of months show we have far more than 12 months in our dat set!
len(md['MONTH_NAME'].value_counts()) # it's probably additional whitespace
md['MONTH_NAME'] = map(lambda x: x.replace(" ", ""), md['MONTH_NAME'])

# Problems with event type names -- sure. Remove repeates.
# Excessive Heat or High Heat? WINTER WEATHER or Winter Weather
md['EVENT_TYPE'] = map(lambda x: x.replace("WINTER WEATHER", "Winter Weather"), md['EVENT_TYPE'])
md['EVENT_TYPE'] = map(lambda x: x.replace("Excessive ", ""), md['EVENT_TYPE'])

# Now convert strings into categorical variables
# md['STATE'] = pd.Categorical(md['STATE']).labels
keepEvent = md['EVENT_TYPE']
md['EVENT_TYPE'] = pd.Categorical(md['EVENT_TYPE']).labels
keepMonths = md['MONTH_NAME']
md['MONTH_NAME'] = pd.Categorical(md['MONTH_NAME']).labels

test_idx = np.random.uniform(0, 1, len(md)) <= 0.3
train = md[test_idx==True]
test = md[test_idx==False]

train_target = train['DAMAGE_PROPERTY_SCALED']
# train_data = train.ix[:,['DURATION', 'EVENT_TYPE', 'STATE_FIPS', 'MONTH_NAME']]
train_data = train.ix[:,['MONTH_NAME']]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfr = RandomForestRegressor(n_estimators=100, oob_score=True)
rfc.fit(train_data, train_target)
rfr.fit(train_data, train_target)

rfc.oob_score_

test_target = test['DAMAGE_PROPERTY_SCALED']
# test_data = test.ix[:,['DURATION', 'EVENT_TYPE', 'STATE_FIPS', 'MONTH_NAME']]
test_data = test.ix[:,['MONTH_NAME']]
test_pred = rfr.predict(test_data)

r2 = r2_score(test_target, test_pred)
mse = np.mean((test_target - test_pred)**2)

# test_cm = skm.confusion_matrix(test_target, test_pred)

print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))
print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))

# # Ideas for improving model
# # 1. Use more predictors?
# # 2. Consider feature engineering -- storm duration, 

# # harmfulData['EVENT_TYPE2'] = pd.Categorical(harmfulData['EVENT_TYPE']).labels
# # test_idx = np.random.uniform(0, 1, len(harmfulData)) <= 0.3
# # train = harmfulData[test_idx==True]
# # test = harmfulData[test_idx==False]
# # train_target = train['EVENT_TYPE2']
# # train_data = train.ix[:,1:-2]
# # model = RandomForestClassifier(n_estimators=500, oob_score=True)
# # model.fit(train_data, train_target)

# ##### PREDICTING PROPERTY DAMAGE OF STORM

# # Predict property damage by storm. For example, what is the expected property damage caused by a flood in 


