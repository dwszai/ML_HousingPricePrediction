#=============================IMPORT LIBRARIES=================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')


"""-----------------------DATA PREPROCESSING SECTION------------------------"""

#=============================IMPORT DATASET===================================
# Train dataset
train = pd.read_csv("C:/Users/User/Desktop/housing_prices/dataset/train.csv")
train.head()
# Test dataset
test = pd.read_csv("C:/Users/User/Desktop/housing_prices/dataset/test.csv")
test.head()

# Describe dataset
print("Train set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))
print("Test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))

# Information about dataset
train_stats = train.describe().T
# Understand each features datatype
train.dtypes
# Display no. of numerical and categorical data types
print("The total number of each data type: \n{}".format(train.get_dtype_counts()))


#===============================TARGET VARIABLE================================
# Target variable (predict): SalePrice
def plot_chart(df, feature):
    """Display histogram, boxplot and QQ plot graph
    
    Args:
        df (str): The dataframe (input dataset)
        feature (str): The target variable, one column header of the dataset
    
    Returns:
        figure: A figure containing 3 plots for visualization  
    """
    
    # Set figure dimension and allocate grids in figure
    fig = plt.figure(figsize=(16,10), constrained_layout=True)
    gs = fig.add_gridspec(3,3)
    # Histogram
    ax1 = fig.add_subplot(gs[0,:2])
    sns.distplot(df.loc[:,feature], ax=ax1).set_title('Histogram')   
    # QQ plot
    ax2 = fig.add_subplot(gs[1,:2])
    stats.probplot(df.loc[:, feature], plot=ax2)
    # Box plot
    ax3 = fig.add_subplot(gs[:,2])
    sns.boxplot(df.loc[:,feature], orient='v', ax=ax3).set_title('Box plot')
    
# Findings: abnormal distribution, right-skewed, outliers present
plot_chart(train, 'SalePrice')

# Check for skewness and kurtosis value, rounded
skewness = train['SalePrice'].skew()
kurtosis = train['SalePrice'].kurt()

print("Skewness: {}".format(round(skewness, 2)))
if skewness > 0:
    print("Positive/right skewness: mean and median > mode.")
else:
    print("Negative/left skewness: mean and median < mode")
    
print("\nKurtosis: {}".format(round(kurtosis, 2)))
if kurtosis > 3:
    print("Leptokurtic: more outliers")
else:
    print("Platykurtic: less outliers")


#=============================FEATURES CORRELATION=============================
# All correlations of features vs target variable in list form
corr_feat = (train.corr())["SalePrice"].sort_values(ascending=False)[1:]

# Correlation matrix in heatmap
sns.set_style('whitegrid')
plt.subplots(figsize = (15,10))
m1 = np.zeros_like(train.corr(), dtype=np.bool)     # Generate mask for upper triangle
m1[np.triu_indices_from(m1)] = True
sns.set(font_scale=0.5)
sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), 
            mask=m1, center = 0, square=True)
plt.title("All features heatmap", fontsize=10)

# Main features focused heatmap
n = 10      # Number of variables
features = train.corr().nlargest(n, 'SalePrice')['SalePrice'].index
hm_data = np.corrcoef(train[features].values.T)
sns.set(font_scale=0.8)
hm = sns.heatmap(hm_data, cbar=True, cmap=sns.diverging_palette(20, 220, n=200), 
                 annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=features.values, xticklabels=features.values)

# Scatter plot features to compare
features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
            'FullBath', 'TotRmsAbvGrd','YearBuilt']
def scatter_plots(df, features):
    """Scatter plots of main features vs target variable
    
    Args:
        df (str): The dataframe (input dataset)
        features (list): A list of feature variables
    
    Returns:
        figure: scatter plots of each feature vs target variable 
    """
    sns.set(style='darkgrid')
    sns.pairplot(df[features])
    
scatter_plots(train, features)


#==============================HANDLE OUTLIERS=================================
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
def bivariate(df, predictor, target):
    data = pd.concat([df[target], df[predictor]], axis=1)
    data.plot.scatter(x=predictor, y=target, ylim=(0, 800000))

# 2nd most correlated (0.71)
bivariate(train, 'GrLivArea', 'SalePrice')
# Delete selected outliers
train = train[train['GrLivArea'] < 4500]
train.reset_index(drop = True, inplace = True)
# Copy for comparison chart
pre_train = train.copy()
# Choose whether to drop, uncertanity
bivariate(train, 'TotalBsmtSF', 'SalePrice')

# Plots to see outliers
train['TotalBsmtSF'].hist(bins=100)
train.boxplot(column=['TotalBsmtSF'])        
train['TotalBsmtSF'].describe()                
print("Max value of TotalBsmtSF: {}\n75 percentile value of TotalBsmtSF: {}\nPossibly outlier".format(train['TotalBsmtSF'].max(), train['TotalBsmtSF'].quantile(0.75)))


#==============================CHECK REGRESSION=================================

def regression_check(df, y, x1, x2): 
    """Find regression in features using scatter plot and regular lines"""
    fig, (ax1, ax2) = plt.subplots(figsize = (16,8), ncols=2, sharey=False)
    # Scatter plot for y vs x1 
    sns.scatterplot(x=df[x1], y=df[y], ax=ax1)
    # Add regression line.
    sns.regplot(x=df[x1], y=df[y], ax=ax1)
    
    # Scatter plot for y vs x2
    sns.scatterplot(x=df[x2], y=df[y], ax=ax2)
    # Add regression line 
    sns.regplot(x=df[x2], y=df[y], ax=ax2)

# Check for regression in features
regression_check(train, 'SalePrice', 'GrLivArea', 'MasVnrArea')

# Find error variance across true line
plt.subplots(figsize = (12,8))
sns.residplot(train['GrLivArea'], train['SalePrice'])

# Transform target variable using numpy.log1p 
train["SalePrice"] = np.log1p(train["SalePrice"])
# Plot newly transformed
plot_chart(train, 'SalePrice')

# Comparing before and after adjusted Saleprice vs feature 
fig, (ax1, ax2) = plt.subplots(figsize=(22, 6), ncols=2, sharey=False, sharex=False)
sns.residplot(x=pre_train['GrLivArea'], y=pre_train['SalePrice'], ax=ax1).set_title('Before')
sns.residplot(x= train['GrLivArea'], y=train['SalePrice'], ax=ax2).set_title('After')


#===============================MISSING VALUES=================================
# Remove id column
train.drop(columns=['Id'],axis=1, inplace=True)
test.drop(columns=['Id'],axis=1, inplace=True)

# Saving the target variable for y train set 
y = train['SalePrice'].reset_index(drop=True)

# Combine train and test datasets together
full_data = pd.concat((train, test)).reset_index(drop=True)
# Remove the target variable 
full_data.drop(['SalePrice'], axis=1, inplace=True)

def list_missing(df):
    """Display features with missing values in a list
    
    Args:
        df (str): The dataframe (input dataset)
    
    Returns:
        list: list of missing amount and percentage belonging to features
    """
    # Total no. and % of missing values, more than 0 missing values
    total = df.isnull().sum().sort_values(ascending=False)[
            df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round((df.isnull().sum() / df.isnull().count() * 100).sort_values(
            ascending=False), 2)[round((df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False), 2) != 0]
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return(missing_data)

# Missing values in list
print("\nTrain data missing values\n")
missing_train = list_missing(train)
print(missing_train)
print("\nTest data missing values")
missing_test = list_missing(test)
print(missing_test)

def heatmap_missing(df):
    """Heatmap showing missing values"""   
    colours = ['#000099', '#ffff00']        # Yellow = missing, Blue = not missing
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colours)).set_title('Missing values')
    
# Missing value heatmap
heatmap_missing(train)
heatmap_missing(test)    


#============================HANDLE MISSING DATA===============================
""" 
    -PoolQC, MiscFeature, Alley, fence, fireplacequ: high missing % and useless info
    -garageXs only 5 percent missing and not related to most impt garageCar variable (0.64)
    -BSMT_Xs only 2 % missing, not related to most impt TotalBsmtSF (0.61)
    -MasVnrType and MasVnrArea unimportant
    -Electrical only 1 missing value, impute value
    => remove the features and impute single observation
"""
missing_data = list_missing(full_data)

# Impute missing value to most frequent value
missing_col_mode = ['Electrical', 'Exterior2nd', 'KitchenQual', 'Exterior1st', 'SaleType']
for col in missing_col_mode:
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])
missing_data = list_missing(full_data)
# Impute missing value to 0   
missing_col_0 = ['GarageCars', 'TotalBsmtSF', 'GarageArea', 'BsmtUnfSF', 
                 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtHalfBath']
for col in missing_col_0:
    full_data[col] = full_data[col].fillna(0)
missing_data = list_missing(full_data)
    
# Drop leftover missing unimportant features with more than 1 missing value
missing_features = missing_data[missing_data['Total'] > 1].index
full_data = full_data.drop(missing_features, 1)
missing_data = list_missing(full_data)

def check_missing(df):
    """Check any missing data left after cleaning"""
    left = df.isnull().sum().max()
    if left == 0:
        print("No missing data")
    else:
        print("Missing data exists")   
    print("Cleaned dataset has {} rows and {} columns.".format(df.shape[0], df.shape[1]))

check_missing(full_data)


#===========================HANDLE UNNECESSARY DATA============================
def repetitive(df): 
    """Find features with above 95% repeated values"""
    total_rows = df.shape[0]  
    for col in df.columns:
        count = df[col].value_counts(dropna=False)
        high_percent = (count/total_rows).iloc[0]      
        if high_percent > 0.95:
            print('{0}: {1:.1f}%'.format(col, high_percent*100))
            print(count)
            print()
# View and understand repetitive reason, if uninformative -> drop
repetitive(full_data)

# CHECK if there are any duplicate rows and drop if exist
data_dup_drop = full_data.drop_duplicates()
print(full_data.shape)
print(data_dup_drop.shape)
print("Number of duplicates dropped: ")
print("Rows: {}".format(full_data.shape[0] - data_dup_drop.shape[0]))
print("Columns: {}".format(full_data.shape[1] - data_dup_drop.shape[1]))
# Update new dataset without any duplicates
full_data = data_dup_drop


#=============================FEATURE ENGINEERING==============================
# Example of skewed feature
sns.distplot(full_data['GrLivArea'])
sns.distplot(full_data['LotArea'])
sns.distplot(train['1stFlrSF'])

def fix_skewness(df):
    """Fix skewness in dataframe
    
    Args:
        df (str): The dataframe (input dataset)
    
    Returns:
        df (str): Fixed skewness dataframe 
    """  
    # Skewness of all numerical features
    num_feat = df.dtypes[df.dtypes != "object"].index
    skewed_num_feat = df[num_feat].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_num_feat[abs(skewed_num_feat) > 0.5].index       # high skewed if skewness above 0.5
    
    for feat in high_skew:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

fix_skewness(full_data)

# Example of feature skewness fixed
sns.distplot(full_data['GrLivArea'])
sns.distplot(full_data['LotArea'])
sns.distplot(full_data['1stFlrSF'])
 
# Create new group features from similar exisiting features
full_data['TotalSF'] = full_data['1stFlrSF'] + full_data['2ndFlrSF'] + full_data['TotalBsmtSF']
full_data['YrBltAndRemod'] = full_data['YearBuilt'] + full_data['YearRemodAdd']
full_data['Total_Bathrooms'] = (full_data['FullBath'] + (0.5*full_data['HalfBath']) +
                               full_data['BsmtFullBath'] + (0.5*full_data['BsmtHalfBath']))
full_data['Total_porchSF'] = (full_data['3SsnPorch'] + full_data['EnclosedPorch'] + 
                             full_data['OpenPorchSF'] + full_data['ScreenPorch'] + 
                             full_data['WoodDeckSF'])

# Create dummy variabes
final_data = pd.get_dummies(full_data).reset_index(drop=True)

def overfit_features(df):
    """Find a list of features that are overfitted"""
    overfit = []
    for col in df.columns:
        counts = df[col].value_counts().iloc[0]
        if counts / len(df)*100 > 99.94:
            overfit.append(col)
    return overfit

# Final X train set and X test set
X = final_data.iloc[:len(y), :]     # take as many rows as y dataset (SalePrice), all columns
X_final_test = final_data.iloc[len(y):, :]    # take the rest of the rows and all columns


print("List of overfitted features: \n{}".format(overfit_features(X)))
X = X.drop(overfit_features(X), axis=1)
print("List of overfitted features: \n{}".format(overfit_features(X_final_test)))
X_final_test = X_final_test.drop(overfit_features(X_final_test), axis=1)


print("List of overfitted features: \n{}".format(overfit_features(final_data)))
final_data = final_data.drop(overfit_features(final_data), axis=1)




"""-------------------------FITTING MODEL SECTION---------------------------"""

# Split into training model on 2/3 of train data (X and y) and validate on (1/3) train data 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Split data into train/test sets, k num of folds, shuffle data before split, set constant random generator 
k_folds = KFold(n_splits=10, shuffle=True, random_state=0)

def rmsle(y, y_pred):
    """root mean squared log error(RMSLE) chosen to scale down outliers, nullify their effects
       -only consider relative error:
       relative error = absolute error(magnitude of error) / exact value (magnitude)
       E.g: 
           y=100, X_pred=90 => RMSLE (cal relative error)=10/100 = 0.1
           Y=10000, X_pred=9000 => RMSLE=1000/10000 = 0.1
       -biased penalty:
       larger penalty for underestimation of value than overestimation
       E.g:
           y=1000, X_pred=600 => RMSLE = 0.51 (underestimation)
           y=1000, X_pred=1400 => RMSLE = 0.33 (overestimation)
           Overestimated sale price: if sell more to earn, if buy more money prepared
           Useful for delivery time regression problem
    """
    return np.sqrt(mean_squared_error(y, y_pred))

def rmse(model, X=X):
    """Root mean squared error"""
    rmse = np.sqrt(-cross_val_score(model, X, y, 
                                    scoring="neg_mean_squared_error", cv=k_folds))
    return rmse

# Assign diff alphas values to find best fit for model
r_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
l_alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]     # values close to 1 better choice based on documentation

# Regularization models (prevent overfitting using penalty on coeff), use pipelines
# Ridge model using pipeline to add robustscaler (scale feature based on percentiles, wont be affected by outliers)
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=r_alphas, cv=k_folds))
# Lasso model
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=l_alphas, max_iter=1e7, cv=k_folds, random_state=0))
# ElasticNet model
elastic_net = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=e_l1ratio, alphas=e_alphas, max_iter=1e7, cv=k_folds))
# Support Vector Regression (SVR) used for working with continuous values (tune para for diff results)
svr = make_pipeline(RobustScaler(), SVR(gamma=0.0003, C=20, epsilon=0.008))    # Small gamma value define a Gaussian function with a large variance

# Light GBM
lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000,
                         max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7,
                         feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)
# XGBoost
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7, colsample_bytree=0.7, objective='reg:linear', nthread=-1, 
                       scale_pos_weight=1, seed=27, reg_alpha=0.00006)
# Emsemble learning, using multiple regressors to predict 
stack_reg = StackingCVRegressor(regressors=(ridge, lasso, elastic_net, svr, lightgbm, xgboost),
                                meta_regressor=xgboost, use_features_in_secondary=True)

# Score of each model
score = rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(elastic_net)
print("Elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(lightgbm)
print("LightGBM: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(xgboost)
print("Xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())

score = rmse(stack_reg)
print("Stack: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now())


# Fit model
print('Fitting Model...')

print('Ridge') 
ridge_model = ridge.fit(X, y)

print('Lasso')
lasso_model = lasso.fit(X, y)

print('ElasticNet')
elastic_model = elastic_net.fit(X, y)

print('SVR')
svr_model = svr.fit(X, y)

print('LightGBM')
lgb_model = lightgbm.fit(X, y)

print('XGBoost')
xgb_model = xgboost.fit(X, y)

print('StackRegressor')
stack_reg_models = stack_reg.fit(np.array(X), np.array(y))

# Blend models
def blend_models_predict(X):
    return ((0.2 * ridge_model.predict(X)) +
            (0.05 * lasso_model.predict(X)) + 
            (0.1 * elastic_model.predict(X)) +    
            (0.1 * svr_model.predict(X)) + 
            (0.1 * lgb_model.predict(X)) + 
            (0.15 * xgb_model.predict(X)) +             
            (0.3 * stack_reg_models.predict(np.array(X))))
    
# Test accuracy score, lower RMSLE better accuracy
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))

# Create new submission file for prediction values
print('Predict submission')
submission = pd.read_csv("C:/Users/User/Desktop/housing_prices/dataset/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_final_test)))
submission.to_csv("submission2.csv", index=False)

"""
print('Submit prediction')
pred_test = np.floor(np.expm1(blend_models_predict(X_final_test)))
submission = pd.DataFrame({'Id': X_final_test.index, 'SalePrice': pred_test})
submission.to_csv('submission.csv', index=False)
"""