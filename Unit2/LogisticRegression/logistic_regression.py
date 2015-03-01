import pandas as pd 
import statsmodels.api as sm 
import collections

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# clean data as necessary
loansData['Interest.Rate'] = map(lambda x: float(x[:x.find('%')]), loansData['Interest.Rate'])

cleanFICO = loansData['FICO.Range'].map(lambda x: x.split('-'))
cleanFICO = cleanFICO.map(lambda x: [int(n) for n in x])
loansData['FICO.Score'] = cleanFICO.map(lambda x: x[0])

# Create break point to serve as interest rate threshold (prob of getting a loan under this threshold)
threshold = 11.3
loansData['High.Rate'] = loansData['Interest.Rate'].map(lambda x: x <= threshold)

# Logit model needs intercept...
loansData['Intercept'] = 1.0

# state the intercept and independant values in list
ind_vars = ['Intercept', 'FICO.Score', 'Amount.Requested']

# Create the model
logit = sm.Logit(loansData['High.Rate'], loansData[ind_vars])
result = logit.fit()
coeff = result.params

def logistic_function(FICO, Amount):
	print "Interest Rate = " + str(threshold)
	print "FICO Score = " + str(FICO)
	print "Amount Requested = " + str(Amount)
	p = (1/(1 + 2.7182**(coeff[0] + coeff[1]*FICO + coeff[2]*Amount)))
	print "Probability of obtaining loan: " + str((1 - p))
	if (p < 0.25):
		print 'Loan Funded. Good day for you!'
	else:
		print 'Loan Not Funded. Increase FICO, request less loot, or increase interest rate offer.'

# need to inverse the result of the logit function to get prob of obtaining loan
# still don't understand linear compoent of function. what does below mean?
# interest_rate = (coeff[0] + coeff[1]*FICO + coeff[2]*Amount)
# what if, given my FICO Score and loan amount, I wanted to know what interest rate I'd have to pay to obtain a loan?
# if that's the linear model.. what does ... hmmm....

# still digging. Using the linear model at 720 and 10,000 expected interest rate is 11.3%
# However, at 11.3%, 720, and 10,000 the loan is not Funded
# Soooo, I am guessing that the linear model needs an additional parameter