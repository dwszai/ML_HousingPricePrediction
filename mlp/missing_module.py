import seaborn as sns
import pandas as pd

# Only use this function if dataset contains missing values for observation
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

# Function to check for missing values in original data or cleaned data (2 usages)
def check_missing(df):
    """Check any missing data"""
    left = df.isnull().sum().max()
    if left == 0:
        print("No missing data")
    else:
        print("Missing data exists")   
    print("Dataset has {} rows and {} columns.".format(df.shape[0], df.shape[1]))

def heatmap_missing(df):
    """Heatmap showing missing values"""   
    colours = ['#000099', '#ffff00']        # Yellow = missing, Blue = not missing
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colours)).set_title('Missing values')