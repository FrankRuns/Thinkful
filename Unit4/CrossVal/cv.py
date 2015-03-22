import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import datasets, svm
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

print 'Geting and cleaning the data...'
df = pd.read_csv(
	filepath_or_buffer = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
	header=None,
	sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(inplace=True)

x_vars = []
for i, row in df.iterrows():
	x_vars.append([row['sepal_len'], row['sepal_wid'], row['petal_len'], row['petal_wid']])

df['class'] = pd.Categorical(df['class']).labels
y_var = list(df['class'])

print 'Splitting data into train and test set...'
x_train, x_test, y_train, y_test = train_test_split(x_vars, y_var, test_size=0.4, random_state=42)

print 'Fitting SVM classifier...'
svc = svm.SVC(kernel='linear')
svc.fit(x_train, y_train)

print 'Predicting values for test set...'
results = svc.predict(x_test)

print 'For each prediction, did we predict correctly?'
for i in range(len(results)):
	if results[i] == y_test[i]:
		print 'Prediction correct'
	else:
		print 'Prediction INcorrect'

print 'Getting accuracy score...'
print accuracy_score(y_test, results)

# TODO: Come back to perform kfold cross validation