import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter
from scipy.cluster.vq import kmeans, vq, whiten

print 'Preparing the data from UCI...'

df = pd.read_csv(
	filepath_or_buffer = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
	header=None,
	sep=',')

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(inplace=True)

print 'Dataframe looks like this...'
print '# of observsations is ' + str(len(df))
print df.head()

print 'Seperate x vars from y vars...'
X = df.ix[:,0:4].values
y = df.ix[:,4].values

# traces = []

# legend = {0:False, 1:False, 2:False, 3:True}

# colors = {'Iris-setosa': 'rgb(31, 119, 180)', 
#           'Iris-versicolor': 'rgb(255, 127, 14)', 
#           'Iris-virginica': 'rgb(44, 160, 44)'}

# for col in range(4):
#     for key in colors:
#         traces.append(Histogram(x=X[y==key, col], 
#                         opacity=0.75,
#                         xaxis='x%s' %(col+1),
#                         marker=Marker(color=colors[key]),
#                         name=key,
#                         showlegend=legend[col]))

# data = Data(traces)

# layout = Layout(barmode='overlay',
#                 xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
#                 xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
#                 xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
#                 xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
#                 yaxis=YAxis(title='count'),
#                 title='Distribution of the different Iris flower features')

# fig = Figure(data=data, layout=layout)
# py.image.save_as({'data': data}, 'iris_hisos.png')
# py.iplot(fig)

print 'Normalize the x vars...'
# standardize data on unit scale
X_std = StandardScaler().fit_transform(X)

print 'Using the x vars, calculate the covariance matrix, eigenvalues / genvectors / eigenpairs and explained variance of each variable. The cumulative explained variance array looks like this:'''
# calculate covariance matrix
# mean_vec = np.mean(X_std, axis=0)
# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# u,s,v = np.linalg.svd(X_std.T)

# selecting features
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()

# determine explained variance of each eigenvalue
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print cum_var_exp

# trace1 = Bar(
# 	x=['PC %s' %i for i in range(1,5)],
# 	y=var_exp,
# 	showlegend=False)

# trace2 = Scatter(
# 	x=['PC %s' %i for i in range(1,5)],
# 	y=cum_var_exp,
# 	name='cumulative explained variance')

# data = Data([trace1, trace2])

# layout = Layout(
# 	yaxis=YAxis(title='Explained variance in percent'),
# 	title='Explained variance by different principle components')

# fig = Figure(data=data, layout=layout)
# py.iplot(fig)

# take the eigenvalues / vectors of our select features
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
					  eig_pairs[1][1].reshape(4,1)))

# go back to usable values
Y = X_std.dot(matrix_w)

print 'Build PCA and fit normalized x vars...'
# PCA with scikit learn
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print Y_sklearn
print len(Y_sklearn)

df['class'] = pd.Categorical(df['class']).labels
first = []; second = []
for i in Y_sklearn:
	first.append(i[0])
for i in Y_sklearn:
	second.append(i[1])
plt.scatter(first, second, c=df['class'])
plt.show()

#################################################

un = pd.read_csv("/Users/frankCorrigan/ThinkfulData/undata.csv")
data = un.dropna()

alt_data = data.ix[:,2:]

std_alt_data = StandardScaler().fit_transform(alt_data)

sklearn_pca = sklearnPCA(n_components=8)

pca_alt_data = sklearn_pca.fit_transform(std_alt_data)

w = whiten(pca_alt_data)

##### Need to know how many components to take...
cov_mat = np.cov(std_alt_data.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

