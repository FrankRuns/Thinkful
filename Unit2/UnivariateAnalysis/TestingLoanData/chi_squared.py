from scipy import stats
import collections

loansData = pd.read_csv("https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv")
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

plt.figure()
plt.bar(freq.keys(), freq.values(), width=1)
plt.show()
# data right skewed. About 40 unique values. Most frequent # is 9.

chi, p = stats.chisquare(freq.values())
# chi = 2408.4331
# p = 0.0

# the chi statistics is large... so this tells me that these observations are unusual, right?
# what's the explanation? -- is it that people on the lending club can't get a loan from a bankso they are attempting peer to peer lending?
# do we get these stats because the observed data is not normally distributed?
# how do we know what the expected distribution for open credit lines is?

