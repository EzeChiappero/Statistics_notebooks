#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


# Load in diamonds data set from seaborn package
diamonds = sns.load_dataset("diamonds", cache=False)

# Examine first 5 rows of data set
diamonds.head()


# In[3]:


sns.histplot(data=diamonds,x='price')


# In[4]:


# Check how many diamonds are each color grade
diamonds["color"].value_counts()


# In[5]:


diamonds.shape


# In[23]:


# Import math package
import math


# In[22]:


# Create boxplot to show distribution of price by cut
sns.boxplot(x = "cut", y = "price", data = diamonds)


# In[24]:


# Create boxplot to show distribution of price by color grade
sns.boxplot(x = "color", y = "price", data = diamonds)


# In[25]:


sns.pairplot(data = diamonds)


# In[8]:


# Import statsmodels and ols function
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[10]:


# Construct simple linear regression model, and fit the model
model = ols(formula = "price ~ C(color)", data = diamonds).fit()


# In[11]:


# Get summary statistics
model.summary()


# In[12]:


# Run one-way ANOVA
sm.stats.anova_lm(model, typ = 2)


# In[13]:


sm.stats.anova_lm(model, typ = 1)


# In[14]:


sm.stats.anova_lm(model, typ = 3)                 


# In[15]:


# Construct a multiple linear regression with an interaction term between color and cut
model2 = ols(formula = "price ~ C(color) + C(cut) + C(color):C(cut)", data = diamonds).fit()


# In[17]:


model2.summary()


# In[18]:


# Run two-way ANOVA
sm.stats.anova_lm(model2, typ = 2)


# In[19]:


sm.stats.anova_lm(model2, typ = 1)


# In[20]:


sm.stats.anova_lm(model2, typ = 3)


# In[26]:


# Import Tukey's HSD function
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[28]:


# Run Tukey's HSD post hoc test for one-way ANOVA
tukey_oneway = pairwise_tukeyhsd(endog = diamonds["price"], groups = diamonds["color"], alpha = 0.05)


# In[29]:


# Get results (pairwise comparisons)
tukey_oneway.summary()


# In[ ]:




