#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import fuzzywuzzy
from fuzzywuzzy import process
import plotly.express as px
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

# Created python modules
from mlp import missing_module as mm
from mlp import visualization_module as vm
from mlp import feature_engineer_module as fm
from mlp import algorithm_module as am

warnings.filterwarnings('ignore')
print("Libraries imported.")


# ### Read train and test datasets

# In[2]:


sample_submission = pd.read_csv("C:/Users/User/Desktop/ML Projects/housing_prices/dataset/sample_submission.csv")
test = pd.read_csv("C:/Users/User/Desktop/ML Projects/housing_prices/dataset/test.csv")
train = pd.read_csv("C:/Users/User/Desktop/ML Projects/housing_prices/dataset/train.csv")


# In[3]:


train_profile = ProfileReport(train)
train_profile


# In[4]:


# Describe dataset
print("Train set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))
print("Test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))

# Information about dataset
train_stats = train.describe().T
# Understand each features datatype
train.dtypes
# Display no. of numerical and categorical data types
print("The total number of each data type: \n{}".format(train.dtypes.value_counts()))


# In[5]:


train.head()


# ### Understand target variable using charts

# In[6]:


# Target variable (predict): SalePrice

# Findings: abnormal distribution, right-skewed, outliers present
vm.plot_chart(train, 'SalePrice')


# > ### Skewness?? Kurtosis value??

# In[7]:


# Check for skewness and kurtosis value, rounded
fm.skew_kurtosis_value(train, 'SalePrice')


# > ### Correlations of all features vs target variable

# In[8]:


# All correlations of features vs target variable in list form
corr_feat = (train.corr())["SalePrice"].sort_values(ascending=False)[1:]
corr_feat


# ### *Let's view it in heatmap!!!*

# In[9]:


# Correlation matrix in heatmap
sns.set_style('whitegrid')
plt.subplots(figsize = (15,10))
m1 = np.zeros_like(train.corr(), dtype=np.bool)     # Generate mask for upper triangle
m1[np.triu_indices_from(m1)] = True
sns.set(font_scale=0.5)
sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), 
            mask=m1, center = 0, square=True)
plt.title("All features heatmap", fontsize=10)


# In[10]:


# Main features focused heatmap
vm.heatmap_focused(train, 'SalePrice', 10)


# ### Scatter plot comparison

# In[11]:


# Scatter plot features to compare
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            'FullBath', 'TotRmsAbvGrd','YearBuilt']    
vm.scatter_plots(train, features)


# ## Outliers?! 

# In[12]:


""" 
    Univariate analysis:
        -establish a threshold to determine an outlier
            => done by standardize data: convert data values to their mean value = 0 and SD = 1
    
    Findings:
        -low range values close to 0
        -high range values far from 0, highest 2 values very far (may be outliers)
"""
# Adjusted sale price, increase array dimension by 1
saleprice_adj = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])
high_range = saleprice_adj[saleprice_adj[:, 0].argsort()][-10:]     # last 10 in range
low_range = saleprice_adj[saleprice_adj[:, 0].argsort()][:10]       # first 10 in range
print('Outer low range of the distribution:')
print(low_range)
print('\nOuter high range of the distribution:')
print(high_range)


# > ### Bivariate analysis 

# In[13]:


"""
    Bivariate analysis findings:
        1. GrLivArea vs SalePrice:
            -2 outliers w/ greater x value does not follow increasing price trend (bottom right)
            -May be agricultural land
            -not representive of typical case
                => remove outliers
            -2 outliers that extends far above sale price w/ greater x value
            -found in univariate analysis 7+ values
            -follow trend so keep for now
        2. TotalBsmtSF vs SalePrice:
            -3 outliers with highest x value more than 3000 but do not follow trend
            -could be due to poor quality
                => may or may not remove outliers      
"""
# Bivariate analysis, scatter plot
# 2nd most correlated (0.71)
vm.bivariate_scatter(train, 'GrLivArea', 'SalePrice')


# In[14]:


# Delete selected outliers
train = train[train['GrLivArea'] < 4500]
train.reset_index(drop=True, inplace=True)
# Copy for comparison chart
pre_train = train.copy()


# In[15]:


# Choose whether to drop, uncertanity
vm.bivariate_scatter(train, 'TotalBsmtSF', 'SalePrice')

# Plots to see outliers
train['TotalBsmtSF'].hist(bins=100)
train.boxplot(column=['TotalBsmtSF'])        
train['TotalBsmtSF'].describe()                
print("Max value of TotalBsmtSF: {}\n75 percentile value of TotalBsmtSF: {}\nPossibly outlier".format(train['TotalBsmtSF'].max(), train['TotalBsmtSF'].quantile(0.75)))


# ### Check for any regression in features

# In[16]:


# Check for regression in features
vm.regression_check(train, 'SalePrice', 'GrLivArea', 'MasVnrArea')


# In[17]:


# Find error variance across true line
vm.error_variance(train, 'GrLivArea', 'SalePrice')


# In[18]:


# Transform target variable using numpy.log1p 
train["SalePrice"] = np.log1p(train["SalePrice"])
# Plot newly transformed
vm.plot_chart(train, 'SalePrice')


# > ### Comparison of before and after transformation

# In[19]:


# Comparing before and after adjusted target vs feature
vm.compare_error_variance(pre_train, train, 'GrLivArea', 'SalePrice')


# ### Missing values????????

# In[20]:


# Remove id column
train.drop(columns=['Id'],axis=1, inplace=True)
test.drop(columns=['Id'],axis=1, inplace=True)

# Saving the target variable for y train set 
y = train['SalePrice'].reset_index(drop=True)

# Combine train and test datasets together
full_data = pd.concat((train, test)).reset_index(drop=True)
# Remove the target variable 
full_data.drop(['SalePrice'], axis=1, inplace=True)


# In[21]:


# Missing values in list
print("\nTrain data missing values\n")
missing_train = mm.list_missing(train)
print(missing_train)
print("\nTest data missing values")
missing_test = mm.list_missing(test)
print(missing_test)


# In[22]:


# Missing value heatmap
mm.heatmap_missing(train)


# In[23]:


mm.heatmap_missing(test)


#  # Findings
# *   PoolQC, MiscFeature, Alley, fence, fireplacequ: high missing % and useless info
# *     GarageXs only 5 percent missing and not related to most impt garageCar variable (0.64)
# *     BSMT_Xs only 2 % missing, not related to most impt TotalBsmtSF (0.61)
# *     MasVnrType and MasVnrArea unimportant
# *     Electrical only 1 missing value, impute value
# *     => remove the features and impute single observation

# In[24]:


missing_data = mm.list_missing(full_data)
missing_data


# In[26]:


# Impute missing value to most frequent value
missing_col_mode = ['Electrical', 'Exterior2nd', 'KitchenQual', 'Exterior1st', 'SaleType']
for col in missing_col_mode:
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])
missing_data = mm.list_missing(full_data)


# In[27]:


# Impute missing value to 0   
missing_col_0 = ['GarageCars', 'TotalBsmtSF', 'GarageArea', 'BsmtUnfSF', 
                 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath']
for col in missing_col_0:
    full_data[col] = full_data[col].fillna(0)
missing_data = mm.list_missing(full_data)


# In[28]:


# Drop leftover missing unimportant features with more than 1 missing value
missing_features = missing_data[missing_data['Total'] > 1].index
full_data = full_data.drop(missing_features, 1)
missing_data = mm.list_missing(full_data)


# In[29]:


mm.check_missing(full_data)


# In[30]:


# Convert these features to categorical
full_data['MSSubClass'] = full_data['MSSubClass'].apply(str)
full_data['OverallCond'] = full_data['OverallCond'].astype(str)
full_data['YrSold'] = full_data['YrSold'].astype(str)
full_data['MoSold'] = full_data['MoSold'].astype(str)


# ### Unnecessary data, drop if needed

# In[31]:


# View and understand repetitive reason, if uninformative -> drop
fm.repetitive(full_data)


# In[32]:


# CHECK if there are any duplicate rows and drop if exist
fm.duplicate_drop(full_data)
# Update new dataset without any duplicates, drop or not depends on requirements
# full_data = data_dup_drop


# # **Feature Engineering**

# In[33]:


# Example of skewed feature
vm.skew_plot(full_data, 'GrLivArea', 'LotArea')


# In[34]:


# Skewness features list
fm.skewness_list(full_data)


# # Fix skewness
# ##### Using boxcox transformation

# In[35]:


fm.fix_skewness(full_data)
print("Skewness fixed")


# In[36]:


# Example of feature skewness fixed
vm.skew_plot(full_data, 'GrLivArea', 'LotArea')


# In[37]:


# Create new group features from similar exisiting features
full_data['TotalSF'] = full_data['1stFlrSF'] + full_data['2ndFlrSF'] + full_data['TotalBsmtSF']
full_data['YrBltAndRemod'] = full_data['YearBuilt'] + full_data['YearRemodAdd']
full_data['Total_Bathrooms'] = (full_data['FullBath'] + (0.5*full_data['HalfBath']) +
                               full_data['BsmtFullBath'] + (0.5*full_data['BsmtHalfBath']))
full_data['Total_porchSF'] = (full_data['3SsnPorch'] + full_data['EnclosedPorch'] + 
                             full_data['OpenPorchSF'] + full_data['ScreenPorch'] + 
                             full_data['WoodDeckSF'])


# # Dummies!

# In[38]:


# Create dummy variabes
final_data = pd.get_dummies(full_data).reset_index(drop=True)


# In[43]:


# Final X train set and X test set
X = final_data.iloc[:len(y), :]     # take as many rows as y dataset (SalePrice), all columns
X_final_test = final_data.iloc[len(y):, :]    # take the rest of the rows and all columns


# > > ### Remove overfit features

# In[44]:


# Decide whether to drop overfitted features
overfits = fm.overfit_features(X)

print("List of overfitted features: \n{}".format(overfits))
X = X.drop(overfits, axis=1)
print("List of overfitted features: \n{}".format(overfits))
X_final_test = X_final_test.drop(overfits, axis=1)
print("Overfit features dropped.")


# # Fit model

# In[45]:


# Split train data into train/test sets, k num of folds, shuffle data before split, set constant random generator 
# Assign diff alphas values to find best fit for model
# Algorithm class object (above all done in algorithm class)
a = am.Algorithms()


# # Regularization models

# In[46]:


# Regularization algorithms
a.regularization_models()

# Regression algorithms
a.regression_models()


# ## Score models

# In[47]:


# Score of each model
score = a.rmseCV(X, y, a.ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.elastic_net)
print("Elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.lightgbm)
print("LightGBM: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = a.rmseCV(X, y, a.xgboost)
print("Xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

#score = a.rmseCV(X, y, a.stack_reg)
#print("Stack: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())


# ## Train models based on above regression techniques

# In[48]:


# Fit model function from algorithm module class
a.fit_models(X, y)


# > ## Blending!!
# > ### Combine to get the best prediction

# In[49]:


# Blend models
a.blend_models_predict(X)


# ### Root mean squared log error(RMSLE) chosen to scale down outliers, nullify their effects
# *        only consider relative error:
# *        relative error = absolute error(magnitude of error) / exact value (magnitude)
# *        E.g: 
# *            y=100, X_pred=90 => RMSLE (cal relative error)=10/100 = 0.1
# *            Y=10000, X_pred=9000 => RMSLE=1000/10000 = 0.1
# *        biased penalty:
# *        larger penalty for underestimation of value than overestimation
# *        E.g:
# *            y=1000, X_pred=600 => RMSLE = 0.51 (underestimation)
# *            y=1000, X_pred=1400 => RMSLE = 0.33 (overestimation)
# *            Overestimated sale price: if sell more to earn, if buy more money prepared
# *            Useful for delivery time regression problem

# ### Test accuracy score

# In[50]:


# Test accuracy score, lower RMSLE better accuracy
print('RMSLE score on train data:')
print(a.rmsle(y, a.blend_models_predict(X)))


# In[52]:


# Create new submission file for prediction values
print('Predict submission')
submission = sample_submission
submission.iloc[:,1] = np.floor(np.expm1(a.blend_models_predict(X_final_test)))
submission.to_csv("final_submission.csv", index=False)

