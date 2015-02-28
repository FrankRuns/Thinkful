# Challenge
# Write a script called "prob_lending_club.py" that reads in the loan data, cleans it, and loads it into a pandas DataFrame. 
# Use the script to generate and save a boxplot, histogram, and QQ-plot for the values in the "Amount.Requested" column. 
# Be able to describe the result and how it compares with the values from the "Amount.Funded.By.Investors". Push your code to 
# Github and enter the link below.

import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats

loansData = pd.read_csv("https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv")

loansData.dropna(inplace=True)

# loansData.boxplot(column = "Amount.Funded.By.Investors")
# plt.show()
loansData.boxplot(column = "Amount.Requested")
plt.show()
# median amount requested is equal to the median amount funded
# minimum amount funded is 0 --- as would be expected

# loadData.hist(column = "Amount.Funded.By.Investors")
# plt.show()
loansData.hist(column = "Amount.Requested")
plt.show()
# data is skewed to the right. Not normally distributed

plt.figure()
# <----- why do we need plt.figure for qqplot... but not .boxplot or .hist? ---->
# graph = stats.probplot(loansData["Amount.Funded.By.Investors"], dist-"norm", plot=plt)
graph = stats.probplot(loansData["Amount.Requested"], dist="norm", plot=plt)
plt.show()
# confirmed that data is not normally distributed