import pandas.io.data
from pandas import Series, DataFrame
import datetime
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

print '~~~~~~~~~~~~ STOCKS ~~~~~~~~~~~~~~'

# get apple stock data
aapl = pd.io.data.get_data_yahoo('AAPL', start = datetime.datetime(2014, 5, 1), end = datetime.datetime(2014, 6, 1))

# look at daily percent change
aapl['Adj Close'].pct_change().head()

# plot the actual price against moving average
aapl['Adj Close'].plot(label='AAPL')
pd.rolling_mean(aapl['Adj Close'], 40).plot(label='mavg')
plt.legend()
# hmm. No line for mavg...

# get multi stock data
df = pandas.io.data.get_data_yahoo(['AAPL', 'GE', 'IBM', 'KO', 'MSFT', 'PEP'], start = datetime.datetime(2010, 1, 1), end = datetime.datetime(2013, 1, 1))['Adj Close']

# get daily % change for each stock
rets = df.pct_change()

# plot PEP vs KO
plt.scatter(rets.PEP, rets.KO)
plt.xlabel('Returns PEP')
plt.ylabel('Returns KO')
plt.title('Returns of Pepsi vs Coke\n')

# calculate a returns correlation matrix
corr = rets.corr()

# visualize returns correlation matrix
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix: Tech Stocks?")

print '~~~~~~~~~~~~ SUNSPOTS ~~~~~~~~~~~~~~'

print sm.datasets.sunspots.NOTE

dta = sm.datasets.sunspots.load_pandas().data

dta[:2]

dta.plot(x="YEAR", y="SUNACTIVITY", figsize=(12,3))

# create and plot a linear model
regr = linear_model.LinearRegression()

# these could be helpful
# dta = log(dta)
# dta = dta.replace([inf, -inf], np.nan)
# dta = dta.dropna
 
years = [ [x] for x in dta["YEAR"].values ]
sunsp = dta["SUNACTIVITY"].values

yearsTrain = years[:250]
yearsTest = years[251:]
sunspTrain = sunsp[:250]
sunspTest = sunsp[251:]

regr.fit(yearsTrain, sunspTrain)

regr.predict(yearsTest)

# plot the linear model predicted values with the actual data
dta.plot(x="YEAR", y="SUNACTIVITY", figsize=(12,3))
plt.plot(years, regr.predict(years), color='red', linewidth=3)

# explore model little further
print ("R-squared:"); regr.score(yearsTest, sunspTest)
# result of -0.12 which means flat line better estimator than this model

# reshape time series. index is years between 1700 and 2008
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta['YEAR']
dta.plot(figsize(12,3));

import pandas.tools.plotting as pdplot

# first do a lag plot which will show relationship with value this period and value in last period
plt.figure(figsize=(12,12))
pdplot.lag_plot(dta)
plt.title("Sunspots this year vs. last year\n")

# second do lag plots for 1-4 periods
plt.figure(figsize=(12,12))

Lags = [1,2,3,4]

plt.subplot(221)
pdplot.lag_plot(dta, lag=Lags[0])
plt.title("Lag = " + str(Lags[0]))

plt.subplot(222)
pdplot.lag_plot(dta, lag=Lags[1])
plt.title("Lag = " + str(Lags[1]))

plt.subplot(223)
pdplot.lag_plot(dta, lag=Lags[2])
plt.title("Lag = " + str(Lags[2]))

plt.subplot(224)
pdplot.lag_plot(dta, lag=Lags[3])
plt.title("Lag = " + str(Lags[3]))

# definitely autocorrelation exists
# ARMIMA model appropriate

fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

