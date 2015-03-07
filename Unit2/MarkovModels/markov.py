import pandas as pd
df = pd.DataFrame({'bear': [0.150, 0.800, 0.050],
				   'bull': [0.900, 0.075, 0.025],
				   'stag': [0.250, 0.250, 0.500]},
				   index = ['bull', 'bear', 'stag'])

print df
print df.dot(df)
print df.dot(df*df)
print df.dot(df*df*df)
print df.dot(df**200)
