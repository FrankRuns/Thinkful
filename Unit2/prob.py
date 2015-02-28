# Challenge
# Write a script called "prob.py" that outputs frequencies, as well as creates and saves a boxplot, a histogram, 
# and a QQ-plot for the data in this lesson. Make sure your plots have names that are reasonably descriptive. 
#Push your code to GitHub and enter the link below.

import numpy as numpy
import scipy.stats as stats
import matplotlib.pyplot as plt 

data = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]

# boxplot
plt.figure()
plt.boxplot(data)
# plt.savefig("boxplot.png")
plt.show()

# histogram
plt.figure()
plt.hist(data, histtype='bar')
# plt.savefig("histogram.png")
plt.show()

# qqplot
plt.figure()
graph3 = stats.probplot(data, dist="norm", plot=plt)
# plt.savefig("qqplot.png")
plt.show()