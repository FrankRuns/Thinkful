# Submission
# Save your version of "linear_regression.py" and push it to Github. Enter the link below.

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import statsmodels.api as sm 

# get the data
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# remove '%' sign from interest rate data
rmPercentSign = lambda x: float(x[:x.find('%')])
loansData['Interest.Rate'] = map(rmPercentSign, loansData['Interest.Rate'])

# remove word 'month' from loan length data
rmWordMonth = lambda x: float(x[:x.find(' ')])
loansData['Loan.Length'] = map(rmWordMonth, loansData['Loan.Length'])

# select first element from range in FICO Range and turn to int
cleanFICO = loansData['FICO.Range'].map(lambda x: x.split('-'))
cleanFICO = cleanFICO.map(lambda x: [int(n) for n in x])
loansData['FICO.Score'] = cleanFICO.map(lambda x: x[0])

# plot FICO Score
plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()
# FICO scores are skewed to the right (similar to open credit lines)

# create scatter plot matrix
plt.figure()
a = pd.scatter_matrix(loansData, alpha = 0.05, figsize=(10,10), diagonal='hist')
plt.show()
# most interesting relationship is between income and FICO scores
# Thinkful points out that loan amount probably impacts interest rate

# build lineal model to predict interest rate
# independent vars = FICO score and loan amount
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# reshape data from df to matrix
y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# combine x columns to create input matrix
x = np.column_stack([x1,x2])

# create model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print 'Coefficients: ', f.params[1:]
print 'Intercept: ', f.params[0]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

print "Final model: 'Interest Rate = " + str(round(f.params[0])) + " + " + str(f.params[1]) + "*fico" + ' + ' + str(f.params[2]) + "*loanamt"

# the p values are less than 0.05 so we can reject the null and assume all relationships are significant
# the r2 = .65 so our model explains roughly 65% of variation in interest rates
# TODO: consider re-hashing with feature selection package in R (stepwise selection)


