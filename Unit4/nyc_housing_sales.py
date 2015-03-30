# Analyze sales using regression with any predictors you feel are relevant. Justify why regression was appropriate to use.
# Visualize the coefficients and fitted model.
# Predict the neighborhood using a k-NN classifier. Be sure to withhold a subset of the data for testing. Find the variables and the k that give you the lowest prediction error.
# Report and visualize your findings.
# Describe any decisions that could be made or actions that could be taken from this analysis.

print 'Importing libs...'
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# data source = https://data.cityofnewyork.us/Housing-Development/Annualized-Rolling-Sales-Update/uzf5-f8n2

print 'Using pandas to load the data...'
# use sales data from 2009
bronx = pd.read_excel('/Users/frankCorrigan/ThinkfulData/NYCHomeSales2009/2009_bronx.xls', skiprows=4)
brooklyn = pd.read_excel('/Users/frankCorrigan/ThinkfulData/NYCHomeSales2009/2009_brooklyn.xls', skiprows=4)
manhattan = pd.read_excel('/Users/frankCorrigan/ThinkfulData/NYCHomeSales2009/2009_manhattan.xls', skiprows=4)
queens = pd.read_excel('/Users/frankCorrigan/ThinkfulData/NYCHomeSales2009/2009_queens.xls', skiprows=4)
statenisland = pd.read_excel('/Users/frankCorrigan/ThinkfulData/NYCHomeSales2009/2009_statenisland.xls', skiprows=4)

print 'Cleaning... drop nas, drop rows where sale price = 0'
all_boroughs = bronx.append([brooklyn, manhattan, queens, statenisland])
all_boroughs.dropna(inplace=True)
all_boroughs = all_boroughs[all_boroughs['SALE PRICE']!= 0]
data = all_boroughs # shorten name for convenience

print 'Building a histogram of sales price...'
# look at distribution of sales price... badly skewed long tail
sales = []
for el in data['SALE PRICE']:
	sales.append(el)
plt.hist(sales, bins=50)
plt.hist(np.log(sales), bins=50)

# linear regression dependent variable = sales price
# if using 2003 - 2009 data I would also predict the # of sales per month via sales date

print 'Looking for relationships in correlation matrix...'
# look at correlation matrix to find strong replationships
numeric_variables = data.ix[:,[0,4,5,10,11,12,13,14,15,16,18]]
numeric_variables.corr()['SALE PRICE']
# shocked at how low correlations are... obviously more going on than this data set will tell us

print 'Building linear model...'
data['Intercept'] = float(1.0)
model = sm.OLS(data['SALE PRICE'], data[['Intercept', 'GROSS SQUARE FEET']])
result = model.fit()
print result.summary()
# Equation: y = 23,010 + 135x

print 'Building linear model #2...'
model = sm.OLS(data['SALE PRICE'], data[['Intercept', 'GROSS SQUARE FEET', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE']])
result = model.fit()
print result.summary()
# As total units increases, sales price falls? This is... interesting.

# Make Borough categorical and define predictor set
data['BOROUGH'] = pd.Categorical(data['BOROUGH']).labels
# cols = [col for col in data.columns if col not in ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'BUILDING CLASS AT TIME OF SALE', 'SALE DATE']]
# cols = [col for col in data.columns if col not in ['BOROUGH']]
# # split data into train and test sets
# x_train, x_test, y_train, y_test = train_test_split(data.ix[:,1:], data['BOROUGH'], test_size=0.4, random_state=42)

sub_data_cols = [col for col in data.columns if col in ['YEAR BUILT', 'ZIP CODE', 'GROSS SQUARE FEET', 'SALE PRICE', 'COMMERCIAL UNITS', 'BOROUGH']]
data2 = data[sub_data_cols]

test_idx = np.random.uniform(0, 1, len(data2)) <= 0.3
train = data2[test_idx==True]
test = data2[test_idx==False]

features = [col for col in data2.columns if col not in ['BOROUGH']]

# fit the knn algorithm to the test data 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train[features], train['BOROUGH'])

# get predictions and find accuracy
preds = neigh.predict(test[features])
accuracy = np.where(preds==test['BOROUGH'],1,0).sum() / float(len(test))

results = []
for n in range(1,51,2):
	clf = KNeighborsClassifier(n_neighbors=n)
	clf.fit(train[features], train['BOROUGH'])
	preds = clf.predict(test[features])
	accuracy = np.where(preds==test['BOROUGH'], 1, 0).sum() / float(len(test))
	print 'Neighbors: %d, Accuracy: %3f' % (n, accuracy)
	results.append([n, accuracy])




