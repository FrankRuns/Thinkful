import numpy as numpy
import scipy.stats as stats
import matplotlib.pyplot as plt 

plt.figure()
test_data = np.random.normal(size=1000)
graph1 = stats.probplot(test_data, dist="norm", plot=plt)
plt.show()

plt.figure()
test_data2 = np.random.uniform(size=1000)
graph2 = stats.probplot(test_data2, dist="norm", plot=plt)
plt.show()

plt.figure()
data = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]
graph3 = stats.probplot(data, dist="norm", plot=plt)
plt.show()