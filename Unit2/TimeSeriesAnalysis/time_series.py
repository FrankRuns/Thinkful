import pandas as pd 
import numpy as np 

import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots

# read data from local file
ts = pd.read_csv('/Users/frankCorrigan/ThinkfulData/LoanStats3c.csv', header=0, low_memory=False)

# create new column for issue date -- formatted as datetime
ts['issue_d_format'] = pd.to_datetime(ts['issue_d'], format='%b-%y')

# set index to that new column
tso = ts.set_index('issue_d_format')

# 
year_month_summary = tso.groupby(lambda x: x.year * 100 + x.month).count()


loan_count_summary = year_month_summary['issue_d']