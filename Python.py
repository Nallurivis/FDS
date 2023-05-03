#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
dataset = pd.read_csv("data2.csv")


# In[2]:


dataset.head(5)
dataset.info()


# In[3]:


w_bar=dataset.mean()
w_bar
new_w = dataset['Weights'].mean()
std= dataset['Weights'].std()
std


# In[4]:


dataset.describe()


# In[15]:


count, bin_edges = np.histogram(dataset["Weights"])

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins
dataset.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
            color="orange")
plt.title('weights of babies')
plt.ylabel('count')
plt.xlabel('Weights')



# In[ ]:
X= dataset[(dataset.Weights>new_w) & (dataset.Weights < 1.2*new_w)]
print(X)


count, bin_edges = np.histogram(X)

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins
X.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
            color="orange")
plt.title('weights of babies')
plt.ylabel('count')
plt.xlabel('Weights')

# to prove it is a noramal distribution
Y= dataset[(dataset.Weights>(new_w-std)) & (dataset.Weights < (new_w+std))]
print(Y)

