# source of data: https://www.dropbox.com/s/hg0tqy6saqeoq0j/ideal_weight.csv?dl=0
# local source: /Users/frankCorrigan/ThinkfulData/ideal_weight.csv

import pandas as pd 
import matplotlib.pyplot as plt
import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("/Users/frankCorrigan/ThinkfulData/ideal_weight.csv", sep=',')

# Remove single quotes from column names
helper = []
for el in data.columns:
	helper.append(el[1:-1])
data.columns = helper

# Remove single quotes from sex observation data
helper = []
for el in data['sex']:
	helper.append(el[1:-1])
data['sex'] = helper

# Graph histogram for ideal weight and actual weight
plt.hist(data['actual'], bins=20, alpha=0.5, label='actual')
plt.hist(data['ideal'], bins=20, alpha=0.5, label='ideal')
plt.legend(loc='upper right')
plt.show()

# Plot distribution of difference in weight
plt.hist(data['diff'], alpha=0.5, label='actual-ideal')

# Map 'sex' to a categorical variable. 0 = female, 1 = male
data['sex'] = pd.Categorical(data['sex']).labels

# Explore sex variable
data.groupby('sex').describe()

# partition train and test data
train_cut = int(len(data) * 0.7)
fortrain = data[:train_cut]
fortest = data[train_cut:]

# declare training target and predictors and fit the model
train_target = fortrain['sex']
train_data = fortrain.ix[:,2:]
clf = GaussianNB()
clf.fit(train_data, train_target)

# predict on test set
test_target = fortest['sex']
test_data = fortest.ix[:,2:]
pred = clf.predict(test_data)

# see how close predictions are to actual value
testlist = list(test_target)
c = 0
for i in range(len(testlist)):
	if testlist[i] == pred[i]:
		c += 1
print "Correctly predicted " + str(c) + " of " + str(len(testlist)) + " values."

# predict sex for actual = 145, ideal = 160, diff = -15
# 0 = female, 1 = male
print(clf.predict([[145,160,-15]]))
# prediction of 1 makes sense.. dude wants more weight

# predict sex for actual = 160, ideal = 145, diff = 15
# 0 = female, 1 = male
print(clf.predict([[160,145,15]]))
# prediction of 0 makes sense, chick wants less weight
