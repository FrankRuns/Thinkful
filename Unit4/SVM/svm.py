from sklearn import datasets, svm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import ipdb

# load data
iris = datasets.load_iris()

# graph sepal_length vs sepal_width
plt.scatter(iris.data[0:100,1], iris.data[0:100,2], c=iris.target[0:100])
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

# fit the svm on sepal_length and sepal_width data
svc = svm.SVC(kernel='linear', C=1e6)
X = iris.data[0:100, 1:3]
y = iris.target[0:100]
svc.fit(X, y)

# visualize the svm results
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
	estimator.fit(X,y)
	x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
	y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
		                 np.linspace(y_min, x_min, 100))
	Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold)
	#plt.axis('tight')
	plt.axis('off')
	plt.tight_layout()
	plt.show()

plot_estimator(svc, X, y)

