import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import plotly.express as px
import pandas as pd

def plot_chart(df, feature):
    """Display histogram, boxplot and QQ plot graph
    
    Args:
        df (str): The dataframe (input dataset)
        feature (str): The target variable or any feature
    
    Returns:
        figure: A figure containing 3 plots for visualization  
    """
    sns.set(style='darkgrid')
    # Set figure dimension and allocate grids in figure
    fig = plt.figure(figsize=(12,8), constrained_layout=True)
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

def heatmap_full(df):
    """Correlation matrix in heatmap"""
    sns.set_style('whitegrid')
    plt.subplots(figsize = (14,8))
    # Generate mask for upper triangle
    m1 = np.zeros_like(df.corr(), dtype=np.bool)     
    m1[np.triu_indices_from(m1)] = True
    sns.set(font_scale=0.8)
    sns.heatmap(df.corr(), cmap=sns.diverging_palette(20, 220, n=200), 
                mask=m1, center=0, square=True, annot=True)
    plt.title("All features heatmap", fontsize=15)
    
def heatmap_focused(df, target, n):
    """Heatmap of main correlated features against target
    
    Args:
        df (str): The dataframe (input dataset)
        target (str): The target variable
        n (int): Number of features
        
    Returns:
        heatmap figure of most correlated features vs target  
    """
    n = n      # Number of variables
    features = df.corr().nlargest(n, target)[target].index
    hm_data = np.corrcoef(df[features].values.T)
    sns.set(font_scale=0.8)
    sns.heatmap(hm_data, cbar=True, cmap=sns.diverging_palette(20, 220, n=200), 
                annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=features.values, xticklabels=features.values).set_title("Main features heatmap")

def scatter_plots(df, features):
    """Scatter plots of main features vs target variable
    
    Args:
        df (str): The dataframe (input dataset)
        features (list): A list of feature variables
    
    Returns:
        figure: scatter plots of each feature vs target variable 
    """
    sns.set()
    sns.pairplot(df[features], size=3)

def bivariate_boxplot(df, predictor, target):
    """Boxplot"""
    data = pd.concat([df[target], df[predictor]], axis=1)
    fig = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(data=data, x=predictor, y=target)

def bivariate_scatter(df, predictor, target): 
    """Scatter plot"""
    fig = px.scatter(df, x=predictor, y=target)
    fig.show()

def regression_check(df, y, x1, x2): 
    """Find regression in features using scatter plot and regular lines"""
    fig, (ax1, ax2) = plt.subplots(figsize=(16,6), ncols=2, sharey=False)
    # Scatter plot for y vs x1 
    sns.scatterplot(x=df[x1], y=df[y], ax=ax1)
    # Add regression line.
    sns.regplot(x=df[x1], y=df[y], ax=ax1)
    
    # Scatter plot for y vs x2
    sns.scatterplot(x=df[x2], y=df[y], ax=ax2)
    # Add regression line 
    sns.regplot(x=df[x2], y=df[y], ax=ax2)

def error_variance(df, predictor, target):
    """Find error variance across true line"""
    plt.subplots(figsize = (10,6))
    sns.residplot(df[predictor], df[target])

def compare_error_variance(df1, df2, predictor, target):  
    """Comparing before and after adjusted target vs feature """
    fig, (ax1, ax2) = plt.subplots(figsize=(22, 6), ncols=2, sharey=False, sharex=False)
    sns.residplot(x=df1[predictor], y=df1[target], ax=ax1).set_title('Before')
    sns.residplot(x=df2[predictor], y=df2[target], ax=ax2).set_title('After')

def skew_plot(df, f1, f2): 
    """Examples of skewed features"""
    fig, (ax1, ax2) = plt.subplots(figsize=(22, 6), ncols=2, sharey=False, sharex=False)
    sns.distplot(df[f1], ax=ax1)
    sns.distplot(df[f2], ax=ax2)
