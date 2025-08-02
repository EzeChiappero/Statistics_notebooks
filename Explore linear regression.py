#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm # Often imported as sm for Q-Q plot


# In[3]:


# 1. Create a sample dataset
# (Replace this with loading your actual dataset, e.g., df = pd.read_csv('your_data.csv'))

np.random.seed(42) # for reproducibility

data = {
    'TV': np.random.rand(100) * 1000,  # Advertising spend on TV
    'Radio': np.random.rand(100) * 500, # Another potential independent variable
    'Newspaper': np.random.rand(100) * 200, # Another potential independent variable
    'Sales': 50 + 0.05 * (np.random.rand(100) * 1000) + 0.8 * np.random.rand(100) * 100 # Dependent variable
}
# Let's adjust 'Sales' to clearly show a dependency on 'TV' for this example
data['Sales'] = 50 + 0.1 * data['TV'] + np.random.normal(0, 10, 100) # Sales = Intercept + TV_coef * TV + error

df = pd.DataFrame(data)

print("Sample DataFrame head:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("-" * 40)


# In[4]:


# 2. Define the OLS formula

# The formula string uses R-style syntax: 'Dependent_Variable ~ Independent_Variable_1 + Independent_Variable_2'
# In your case: 'Sales ~ TV'
formula = 'Sales ~ TV'

print(f"OLS Formula defined: '{formula}'")
print("-" * 40)


# In[5]:


# 3. Create the OLS model instance

# smf.ols() takes the formula string and your DataFrame as input.
# This creates the *model object* (the blueprint instance), but it's not yet "trained."
model_instance = smf.ols(formula, data=df)

print("OLS model instance created.")
print("It's currently an 'untrained' model object, ready to learn.")
print("-" * 40)


# In[6]:


model_instance.summary()


# In[7]:


# 4. Fit the OLS model

# The .fit() method performs the actual regression calculations.
# It computes the coefficients, standard errors, R-squared, etc.
# It returns the "trained" model results object.
results = model_instance.fit()

print("OLS model has been fitted (trained).")
print("The 'results' object now contains all the calculated regression statistics.")
print("-" * 40)


# In[8]:


# 5. Display the model summary

# The .summary() method provides a comprehensive report of the regression results.
print("OLS Model Summary:")
print(results.summary())


# In[13]:


results.summary()


# In[10]:


print(model_instance)


# In[11]:


# --- Get the residuals from the fitted model ---
residuals = results.resid

print("Residuals head:")
print(residuals.head())
print("-" * 40)


# In[12]:


# --- Create the plots to check for normality ---

plt.figure(figsize=(14, 6)) # Adjust figure size for two plots

# Plot 1: Histogram of Residuals
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
sns.histplot(residuals, color='skyblue', bins=20) # kde=True adds a Kernel Density Estimate line
plt.title('Histogram of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Q-Q Plot of Residuals
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
sm.qqplot(residuals, line='s', ax=plt.gca()) # 's' for standardized line
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout() # Adjusts plot parameters for a tight layout
plt.show()

print("\n--- Interpreting the plots ---")
print("1. Histogram of Residuals:")
print("   - Ideally, the histogram should roughly resemble a bell curve (normal distribution), centered around zero.")
print("   - The KDE line should follow the shape of the bars.")
print("2. Q-Q Plot of Residuals (Quantile-Quantile Plot):")
print("   - This plot compares the quantiles of your residuals against the quantiles of a theoretical normal distribution.")
print("   - For normally distributed residuals, the points should fall approximately along the 45-degree straight line.")
print("   - Deviations from the line (especially at the tails) suggest departures from normality.")


# In[ ]:




