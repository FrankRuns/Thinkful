import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 

print 'Preparing the dataset...'

# TODO: load data directly from website
loans = pd.read_csv("/Users/frankCorrigan/ThinkfulData/LoanStats3c.csv")

# subset a smaller DF of only needed data
loansDF = pd.DataFrame(columns = ['InterestRate', 'Income', 'HomeOwn'])
loansDF['InterestRate'] = loans.int_rate
loansDF['Income'] = loans.annual_inc.astype(float)
loansDF['HomeOwn'] = loans.home_ownership

# drop any rows with missing values
loansDF.dropna(inplace=True)

# reformat interest rate. drop '%' sign and convert to float
loansDF['InterestRate'] = map(lambda x: float(x[:x.find('%')]), loansDF['InterestRate'])

print 'Building model #1...'

# include intercept in data
loansDF['Intercept'] = float(1.0)

# fit the model
model = sm.OLS(loansDF['InterestRate'], loansDF[['Intercept', 'Income']])
result = model.fit()

print '~~~~~~ MODEL 1 ~~~~~~' 
print result.summary()

# Add homeownership to model
loansDF['HomeOwn'] = pd.Categorical(loansDF.HomeOwn).labels

model2 = sm.OLS(loansDF['InterestRate'], loansDF[['Intercept', 'Income', 'HomeOwn']])
result2 = model2.fit()

print '~~~~~~ MODEL 2 ~~~~~~' 
print result2.summary()

# Add interaction term between income and home ownsership
loansDF['Interaction'] = loansDF['Income'] * loansDF['HomeOwn']

model3 = sm.OLS(loansDF['InterestRate'], loansDF[['Intercept', 'Income', 'HomeOwn', 'Interaction']])
result3 = model3.fit()

print '~~~~~~ MODEL 3 ~~~~~~' 
print result3.summary()

# # of questions about this process?
# why do we need to manually add the intercept term?
# if home ownership is a categorical variable... how do we interpret that coefficient?
# if we use an interaction term... is it then OK to interpret the coefficients on the individual variables as accurate?
# why have we not addressed p-values, t-statistics, or confidence intervals?
