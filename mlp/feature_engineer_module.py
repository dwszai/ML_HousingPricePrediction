from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import fuzzywuzzy

def skew_kurtosis_value(df, feature):
    """ Function for skewness and kurtosis value
    
    Args:
        df (str): The dataframe (input dataset)
        feature (str): The target variable or any feature
    
    Returns:
        Skewness and kurtosis values
    """
    skewness = df[feature].skew()
    kurtosis = df[feature].kurt()

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

def duplicate_drop(df): 
    # Check for any duplicates left
    data_dup_drop = df.drop_duplicates()
    print(df.shape)
    print(data_dup_drop.shape)
    print("Number of duplicates dropped: ")
    print("Rows: {}".format(df.shape[0] - data_dup_drop.shape[0]))
    print("Columns: {}".format(df.shape[1] - data_dup_drop.shape[1]))

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

def skewness_list(df):  
    num_feat = df.dtypes[df.dtypes != "object"].index
    skewed_num_feat = df[num_feat].apply(lambda x: skew(x)).sort_values(ascending=False)
    return skewed_num_feat

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
    
    # Use boxocx transformation to fix skewness
    for feat in high_skew:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

def inconsistent_feature_check(df, feature):
    print(df[feature].value_counts())
    # View all the different classes
    feature = df[feature].unique()
    feature.sort()
    return feature

def capitalization_fix(df, feature):
    # Change everything to lower case for consistency and precised value placement
    df[feature] = df[feature].str.lower()
    # Remove trailing whitespaces (in case)
    df[feature] = df[feature].str.strip()
    
def replace_matches(df, feature, class_to_match, min_ratio):
    # List of classes in feature
    list_class = df[feature].unique() 
    # Top 10 closest matches
    matches = fuzzywuzzy.process.extract(class_to_match, list_class, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    # Matches with a high ratio (set by user)
    close_matches = [m[0] for m in matches if m[1] >= min_ratio]
    # Rows of all the close matches in our dataframe
    rows_matches = df[feature].isin(close_matches)
    # Replace all rows with close matches with the input matches 
    df.loc[rows_matches, feature] = class_to_match
    print("REPLACED!")

def overfit_features(df):
    """Find a list of features that are overfitted"""
    overfit = []
    for col in df.columns:
        counts = df[col].value_counts().iloc[0]
        if counts / len(df)*100 > 99.94:
            overfit.append(col)
    return overfit
