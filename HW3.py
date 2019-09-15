#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[59]:


df = pd.read_csv('corporate_bond.csv')
df.head()


# In[60]:


df.info()
df.dtypes


# In[8]:


df.describe()


# In[10]:


print(df.shape)


# In[61]:


print(df.describe(percentiles=[0.25,0.5,0.75]))
print(data.tail())


# In[13]:


#Quantile-Quantile Plot
volume = df['weekly_median_volume']
stats.probplot(volume, dist = "norm",plot = plt)
plt.show()
ntrades = df['weekly_median_ntrades']
stats.probplot(volume,dist = "norm",plot = plt)
plt.show()


# In[74]:


#swarm plot
_ = sns.swarmplot('Coupon','LiquidityScore',data = df)
plt.show()


# In[30]:


#Histogram 
sns.set()
n_data = len(df['n_days_trade'])
n_bins = np.sqrt(n_data)
n_bins = int(n_bins)
_ = plt.hist(df['n_days_trade'],bins = n_bins)
_= plt.xlabel('N days trade ')
_ = plt.ylabel('Count')
plt.show()


# In[56]:


#ECDF 
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1,1+n)/n
    return x,y
x_client, y_client = ecdf(df['Client_Trade_Percentage'])
_ = plt.plot(x_client,y_client,marker = '.',linestyle = 'none')
_ = plt.ylabel('ECDF')
_ = plt.xlabel('Client trade percentage')
plt.show()

x_maturity, y_maturity = ecdf(df['Maturity At Issue months'])
_ = plt.plot(x_maturity,y_maturity,marker = '.',linestyle = 'none')
_ = plt.ylabel('ECDF')
_ = plt.xlabel('Maturity At Issue months')
plt.show()

x_trades, y_trades = ecdf(df['n_trades'])
_ = plt.plot(x_trades,y_trades,marker = '.',linestyle = 'none')
_ = plt.ylabel('ECDF')
_ = plt.xlabel('number of trades')
plt.show()


# In[53]:


#Parallel Coordinates Plots
from pandas.plotting import parallel_coordinates
features = df[['Issue Date', 'Maturity', 'S_and_P', 'Moodys', 'Fitch']][:15]
parallel_coordinates(features, 'Moodys')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=2,fancybox=True,shadow=True)
plt.show()


# In[57]:


#Cross-plot
plt.scatter(df["n_trades"], df["LIQ SCORE"])
plt.xlabel('n_trades')
plt.ylabel('LIQ SCORE')
plt.show()


# In[66]:


corMat = pd.DataFrame(data.corr())
corMat


# In[64]:


#Heat map
correlation_matrix = pd.DataFrame(data.corr())
sns.heatmap(correlation_matrix)

