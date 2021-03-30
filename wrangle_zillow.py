import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from env import host, user, password


'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'





def zillow17():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with:
    - all fields related to the properties that are available
    - using all the tables in the database
    - Only include properties with a transaction in 2017
    - include only the last transaction for each property
    - zestimate error
    - date of transaction
    - Only include properties that include a latitude and longitude value
    '''
    
    query = """
    select *
    from properties_2017 as prop17
    join (
    select parcelid, max(logerror) as zestimateerror, max(transactiondate) as transactiondate
    from predictions_2017
    group by parcelid
    order by parcelid
    ) as pred17 using (parcelid)
    left join airconditioningtype
    ON prop17.airconditioningtypeid=airconditioningtype.airconditioningtypeid
    left join architecturalstyletype
    ON prop17.architecturalstyletypeid=architecturalstyletype.architecturalstyletypeid
    left join buildingclasstype
    ON prop17.buildingclasstypeid=buildingclasstype.buildingclasstypeid
    left join heatingorsystemtype
    ON prop17.heatingorsystemtypeid=heatingorsystemtype.heatingorsystemtypeid
    left join propertylandusetype
    ON prop17.propertylandusetypeid=propertylandusetype.propertylandusetypeid
    left join storytype
    ON prop17.storytypeid=storytype.storytypeid
    left join typeconstructiontype
    ON prop17.typeconstructiontypeid=typeconstructiontype.typeconstructiontypeid
    join unique_properties using(parcelid)
    where latitude IS NOT NULL
    and longitude IS NOT NULL
    and transactiondate like '2017%'
    """
    return pd.read_sql(query, get_connection('zillow'))





def missing_values(df):
    """
    missing_values takes in a datatframeof observations and attributes and
    returns a new dataframe consisting of:
    - 1st col: number of rows with missing values for that attribute
    - 2nd col: percent of total rows that have missing values for that attribute
    """
    rows = pd.DataFrame(df.isnull().sum())
    rows = rows.rename(columns={0: "No.MissingValues"})
    
    percent = pd.DataFrame(df.isnull().sum()/len(df)*100)
    percent = percent.rename(columns={0: "PercentMissing"})
    
    df = pd.concat([rows, percent], axis=1)
    
    
    return df





def missing_col_values(df):
    """
    missing_col_values takes in a datatframe of observations and attributes and
    returns a new dataframe consisting of:
    - the number of columns missing
    - percent of columns missing
    - number of rows with n columns missing.
    """
    col1 = pd.DataFrame(df.isna().sum(axis=1))
    col1 = col1.rename(columns={0: "num_cols_missing"})
    
    col2 = pd.DataFrame(((df.isna().sum(axis=1)) / len(df.columns))*100)
    col2 = col2.rename(columns={0: "pct_cols_missing"})
    
    col12 = col1.merge(col2, left_index=True, right_index=True, how='left')
    col12 = col12.sort_values(by=['num_cols_missing'])
    col12.reset_index(drop=True, inplace=True)
    
    col3 = pd.DataFrame(col12.num_cols_missing.value_counts(ascending=True))
    col3 = col3.rename(columns={'num_cols_missing': 'num_rows'})

    
    df1 = pd.merge(col12, col3, left_on='num_cols_missing', right_index=True, how='left')
    
    return df1





'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''

def drop_based_on_pct(df, pc, pr):
    """
    drop_based_on_pct takes in: 
    - dataframe, 
    - threshold percent of non-null values for columns(# between 0-1), 
    - threshold percent of non-null values for rows(# between 0-1)
    
    Returns: a dataframe with the columns and rows dropped as indicated.
    """
    
    tpc = 1-pc
    tpr = 1-pr
    
    df.dropna(axis = 1, thresh = tpc * len(df.index), inplace = True)
    
    df.dropna(axis = 0, thresh = tpr * len(df.columns), inplace = True)
    
    return df
    
    
    


def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df





def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df





def wrangle_zillow():
    df = pd.read_csv('zillow17.csv')
    
    df = df.set_index("parcelid")
    
    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull()) & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = drop_based_on_pct(df, .5, .7)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))
    
    # drop unnecessary columns
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       , 'heatingorsystemtypeid.1', 'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc'])
    
    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7314, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df





def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test





def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df
    
    


    
def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound