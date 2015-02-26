import pandas as pd
from scipy import stats

# Average weekly household spending (in British pounds) on alcohol and tobacco products
data = '''Region, Alcohol, Tobacco
North, 6.47, 4.03
Yorkshire, 6.13, 3.76
Northeast, 6.19, 3.77
East Midlands, 4.89, 3.34
West Midlands, 5.63, 3.47
East Anglia, 4.52, 2.92
Southeast, 5.89, 3.20
Southwest, 4.79, 2.71
Wales, 5.27, 3.53
Scotland, 6.08, 4.51
Northern Ireland, 4.02, 4.56'''

# split long data string into seperate lines
data = data.splitlines()

# split data on each line by comma
data = [i.split(', ') for i in data]

# define features and observations
column_names = data[0]
data_rows = data[1::]

# create dataframe
df = pd.DataFrame(data_rows, columns = column_names)

df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)

print 'The mean for the Alcohol and Tobacco dataset is ' + str(round(df['Alcohol'].mean(), 2)) + ' and ' + str(round(df['Tobacco'].mean(), 2)) + ' respectively.' + '\n'

print 'The median for the Alcohol and Tobacco dataset is ' + str(round(df['Alcohol'].median(), 2)) + ' and ' + str(round(df['Tobacco'].median(), 2)) + ' respectively.' + '\n'

print 'The mode for the Alcohol and Tobacco dataset is ' + str(round(stats.mode(df['Alcohol'])[0], 2)) + ' and ' + str(round(stats.mode(df['Tobacco'])[0], 2)) + ' respectively.' + '\n'

print 'The range for the Alcohol and Tobacco dataset is ' + str(round(max(df['Alcohol']) - min(df['Alcohol']), 2)) + ' and ' + str(round(max(df['Tobacco']) - min(df['Tobacco']), 2)) + ' respectively.' + '\n'

print 'The standard deviation for the Alcohol and Tobacco dataset is ' + str(round(df['Alcohol'].std(), 3)) + ' and ' + str(round(df['Tobacco'].std(), 3)) + ' respectively.' + '\n'

print 'The variance for the Alcohol and Tobacco dataset is ' + str(round(df['Alcohol'].var(), 3)) + ' and ' + str(round(df['Tobacco'].var(), 3)) + ' respectively.' + '\n'

