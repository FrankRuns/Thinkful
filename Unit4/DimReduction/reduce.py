import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
import sklearn.metrics as skm

# work sheet for unit got crazy. too much to do.
# goal here is to run PCA on UCI HAR dataset, then predict activity with random forest

# subjects tells us which subject # was performing the observation
subjects = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/subject_train.txt", header=None, delim_whitespace=True, index_col=False)
subjects.columns = ['Subject']
subjects = pd.DataFrame(subjects)
# feature_names tells us the ... feature names
feature_names = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/features.txt", header=None, delim_whitespace=True, index_col=False)
# x_vars gives all columns of independent variables
x_vars = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/X_train.txt", header=None, delim_whitespace=True, index_col=False)
x_vars.columns = feature_names
# y_vars gives column of activities being performed for each obs
y_var = pd.read_csv("/Users/frankCorrigan/ThinkfulData/UCI HAR Dataset/train/y_train.txt", header=None, delim_whitespace=True, index_col=False)
y_var = pd.Categorical(y_var['Activity']).labels
y_var = pd.DataFrame(y_var)
y_var.columns = ['Activity']

# Let's do PCA
alt = x_vars # im going to include subject for now

# Scale the data
std = StandardScaler().fit_transform(alt)

# Find best number of components
cov_mat = np.cov(alt.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
cum_var_exp[178]

# PCA with scikit learn
sklearn_pca = sklearnPCA(n_components=75)
Y_sklearn = sklearn_pca.fit_transform(alt)

# Bring Activity and Subjects back into the picture
df = pd.DataFrame(Y_sklearn)
df = pd.merge(y_var, df, left_index=True, right_index=True)
df = pd.merge(df, subjects, left_index=True, right_index=True)

# Split data so we can train and test
fortrain = df.query('Subject >= 27')
fortest = df.query('Subject <= 6')
forval = df.query("(Subject >= 21) & (Subject < 27)")

# fit random forest model
train_target = fortrain['Activity']
train_data = fortrain.ix[:,1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

# determine oob to show accuracy of model
rfc.oob_score_

# define test set and make predictions
test_target = fortest['Activity']
test_data = fortest.ix[:,1:-2]
test_pred = rfc.predict(test_data)

print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))

# visualize confusion matrix
test_cm = skm.confusion_matrix(test_target, test_pred)
pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()
