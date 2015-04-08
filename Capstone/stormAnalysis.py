import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm

data = pd.read_csv('allData.csv', low_memory=False)

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
harmfulData = mini_data = data.ix[:,['EVENT_TYPE', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS']]
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
mini_data = data.ix[:,['EVENT_TYPE', 'MONTH_NAME', 'STATE', 'INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'SOURCE']]
mini_data.dropna(inplace=True)
removeKMBT(mini_data, 'DAMAGE_PROPERTY')
removeKMBT(mini_data, 'DAMAGE_CROPS')

mini_data['EVENT_TYPE2'] = pd.Categorical(mini_data['EVENT_TYPE']).labels
keepStates = mini_data['STATE']
mini_data['STATE'] = pd.Categorical(mini_data['STATE']).labels
keepMonths = mini_data['MONTH_NAME']
mini_data['MONTH_NAME'] = pd.Categorical(mini_data['MONTH_NAME']).labels

test_idx = np.random.uniform(0, 1, len(mini_data)) <= 0.3
train = mini_data[test_idx==True]
test = mini_data[test_idx==False]

train_target = train['EVENT_TYPE2']
train_data = train.ix[:,1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

rfc.oob_score_

test_target = test['EVENT_TYPE2']
test_data = test.ix[:,1:-2]
test_pred = rfc.predict(test_data)

test_cm = skm.confusion_matrix(test_target, test_pred)

print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))
print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))

# Ideas for improving model
# 1. Use more predictors?
# 2. Consider feature engineering -- storm duration, 

# harmfulData['EVENT_TYPE2'] = pd.Categorical(harmfulData['EVENT_TYPE']).labels
# test_idx = np.random.uniform(0, 1, len(harmfulData)) <= 0.3
# train = harmfulData[test_idx==True]
# test = harmfulData[test_idx==False]
# train_target = train['EVENT_TYPE2']
# train_data = train.ix[:,1:-2]
# model = RandomForestClassifier(n_estimators=500, oob_score=True)
# model.fit(train_data, train_target)

##### PREDICTING PROPERTY DAMAGE OF STORM

# Predict property damage by storm. For example, what is the expected property damage caused by a flood in 


