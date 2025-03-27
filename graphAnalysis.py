import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("Salaries.csv")
euro = pd.read_csv("Euro12.csv")
auto = pd.read_csv("Automobile_data.csv")

# plt.hist(df['salary'],bins=8, density=1)
# sns.distplot(df['salary'])
# df.groupby(['rank'])['salary'].count().plot(kind='bar')
# sns.set_style("whitegrid")
# sns.barplot(x='rank',y ='salary', data=df, estimator=len)
# ax = sns.barplot(x='rank',y ='salary', hue='sex', data=df, estimator=len)
# sns.violinplot(x = "salary", data=df)
# sns.jointplot(x='service', y='salary', data=df)
# sns.regplot(x='service', y='salary', data=df)
# sns.boxplot(x='rank',y='salary', data=df)
# sns.boxplot(x='rank',y='salary', data=df, hue='sex')
# sns.swarmplot(x='rank',y='salary', data=df)
# sns.pairplot(df)
numberdf = df.select_dtypes(include=['number'])
# print(numberdf.cov())
print(numberdf.corr())
# sns.heatmap(numberdf.corr())
# sns.barplot(x='Team',y='Shots on target', data=euro)
# sns.boxplot(x='company',y='price', data=auto)
plt.show()