import pandas as pd
import re
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

# data source: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/
# local file location for training: /Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train

subjects = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/subject_train.txt", header=None, delim_whitespace=True, index_col=False)
subjects.columns = ['Subject']
len(subjects.stack().value_counts())
# 21 participants, # obs varies for each participant betwee 281 and 409
# 7352 total obs
feature_names = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/features.txt", header=None, delim_whitespace=True, index_col=False)
len(feature_names)
# 561 features
x_vars = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/X_train.txt", header=None, delim_whitespace=True, index_col=False)
helper = []; helper2 = []; helper3 = []; helper4 = []; helper5 = []; helper6 = []
for el in feature_names[1]:
	helper.append(re.sub('[()-]', '', el))
for el in helper:
	helper2.append(re.sub('[,]', '_', el))
for el in helper2:
	helper3.append(el.replace('Body', ''))
for el in helper3:
	helper4.append(el.replace('Mag', ''))
for el in helper4:
	helper5.append(el.replace('mean', 'Mean'))
for el in helper5:
	helper6.append(el.replace('std', 'STD'))
x_vars.columns = helper6
# 7352 observations, 561 columns
y_var = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/y_train.txt", header=None, delim_whitespace=True, index_col=False)
y_var.columns = ['Activity']
# 7352 observations
# 1 = Walking, 2 = Walking Upstairs, 3 = Walking Downstairs, 4 = Sitting, 5 = Standing, 6 = Laying

data = pd.merge(y_var, x_vars, left_index=True, right_index=True)
data = pd.merge(data, subjects, left_index=True, right_index=True)
# 7352 obs, 562 columns

# change activity to categorical variable
data['Activity'] = pd.Categorical(data['Activity']).labels

# partition data into training, test, and validation sets
fortrain = data.query('Subject >= 27')
fortest = data.query('Subject <= 6')
forval = data.query("(Subject >= 21) & (Subject < 27)")

# fit random forest model
train_target = fortrain['Activity']
train_data = fortrain.ix[:,1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

# determine oob to show accuracy of model
rfc.oob_score_

# determine most important features
fi = enumerate(rfc.feature_importances_)
cols = train_data.columns
[(value,cols[i]) for (i,value) in fi if value > 0.04]

# define validation set and make predictions
val_target = forval['Activity']
val_data = forval.ix[:1,-2]
val_pred = rfc.predict(val_data)

# define test set and make predictions
test_target = fortest['Activity']
test_data = fortest.ix[:,1:-2]
test_pred = rfc.predict(test_data)

print("mean accuracy score for validation set = %f" %(rfc.score(val_data, val_target)))
print("mean accuracy score for test set = %f" %(rf.score(test_data, test_target)))

test_cm = skm.confusion_matrix(test_target, test_pred)
pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))
print("Recall = %f" %(skm.recall_score(test_target, test_pred)))
print("F1 score = %f" %(skm.f1_score(test_target, test_pred)))