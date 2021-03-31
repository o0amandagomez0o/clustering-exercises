import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from env import host, user, password
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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
    """
    
    """
    
    
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
    df['county'] = np.where(df.fips == 6037, 0,
                           np.where(df.fips == 6059, 1, 
                                   2))
    
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
    
    # create column for age of home
    df['home_age'] = 2021 - df.yearbuilt
    
    #list of cols to convert to 'int'
    cols = ['fips', 'buildingqualitytypeid', 'bedroomcnt', 'roomcnt', 'home_age', 'yearbuilt', 'assessmentyear', 'regionidcounty', 'regionidzip', 'unitcnt', 'home_age']
    #loop through cols list in conversion
    for col in cols:
        df[col] = df[col].astype('int')
    
    # rename a few cols
    df = df.rename(columns={"propertylandusetypeid.1": "propertylandusetypeid", "bathroomcnt": "bathrooms", "bedroomcnt": "bedrooms", "taxvaluedollarcnt": "home_value"})
    
    # create a categorical version of target by splitting into quartiles
    df['zerror_qrtls'] = pd.qcut(df.zestimateerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    
    return df






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





def split_zillow(df):
    """
    split_zillow will take one argument(df) and 
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """  
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210)
    return train, validate, test




def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train, validate, test = split_zillow(df)
        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test





def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols





def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled




def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array((df.dtypes == "object") | (df.dtypes == "category"))

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols





def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            # stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       # stratify=train_validate[target]
                                      )
    return train, validate, test





def scale_my_data(train, validate, test):
    scaler = StandardScaler()
    scaler.fit(train[['bathrooms', 'calculatedfinishedsquarefeet', 'latitude',
              'longitude', 'lotsizesquarefeet', 'rawcensustractandblock', 
              'regionidcity', 'roomcnt','bedrooms', 'home_age', 'yearbuilt', 
              'structuretaxvaluedollarcnt', 'home_value', 'assessmentyear', 
              'landtaxvaluedollarcnt', 'taxamount']])
    X_train_scaled = scaler.transform(train[['bathrooms', 'calculatedfinishedsquarefeet', 'latitude',
              'longitude', 'lotsizesquarefeet', 'rawcensustractandblock', 
              'regionidcity', 'roomcnt','bedrooms', 'home_age', 'yearbuilt', 
              'structuretaxvaluedollarcnt', 'home_value', 'assessmentyear', 
              'landtaxvaluedollarcnt', 'taxamount']])
    X_validate_scaled = scaler.transform(validate[['bathrooms', 'calculatedfinishedsquarefeet', 'latitude',
              'longitude', 'lotsizesquarefeet', 'rawcensustractandblock', 
              'regionidcity', 'roomcnt','bedrooms', 'home_age', 'yearbuilt', 
              'structuretaxvaluedollarcnt', 'home_value', 'assessmentyear', 
              'landtaxvaluedollarcnt', 'taxamount']])
    X_test_scaled = scaler.transform(test[['bathrooms', 'calculatedfinishedsquarefeet', 'latitude',
              'longitude', 'lotsizesquarefeet', 'rawcensustractandblock', 
              'regionidcity', 'roomcnt','bedrooms', 'home_age', 'yearbuilt', 
              'structuretaxvaluedollarcnt', 'home_value', 'assessmentyear', 
              'landtaxvaluedollarcnt', 'taxamount']])

    train[['bathrooms_scaled', 'calculatedfinishedsquarefeet_scaled', 'latitude_scaled',
              'longitude_scaled', 'lotsizesquarefeet_scaled', 'rawcensustractandblock_scaled', 
              'regionidcity_scaled', 'roomcnt_scaled','bedrooms_scaled', 'home_age_scaled', 'yearbuilt_scaled', 
              'structuretaxvaluedollarcnt_scaled', 'home_value_scaled', 'assessmentyear_scaled', 
              'landtaxvaluedollarcnt_scaled', 'taxamount_scaled']] = X_train_scaled
    validate[['bathrooms_scaled', 'calculatedfinishedsquarefeet_scaled', 'latitude_scaled',
              'longitude_scaled', 'lotsizesquarefeet_scaled', 'rawcensustractandblock_scaled', 
              'regionidcity_scaled', 'roomcnt_scaled','bedrooms_scaled', 'home_age_scaled', 'yearbuilt_scaled', 
              'structuretaxvaluedollarcnt_scaled', 'home_value_scaled', 'assessmentyear_scaled', 
              'landtaxvaluedollarcnt_scaled', 'taxamount_scaled']] = X_validate_scaled
    test[['bathrooms_scaled', 'calculatedfinishedsquarefeet_scaled', 'latitude_scaled',
              'longitude_scaled', 'lotsizesquarefeet_scaled', 'rawcensustractandblock_scaled', 
              'regionidcity_scaled', 'roomcnt_scaled','bedrooms_scaled', 'home_age_scaled', 'yearbuilt_scaled', 
              'structuretaxvaluedollarcnt_scaled', 'home_value_scaled', 'assessmentyear_scaled', 
              'landtaxvaluedollarcnt_scaled', 'taxamount_scaled']] = X_test_scaled
    return train, validate, test






def split_zillowdf(df):
    '''
    dummy var for gender into is_male
    add 'spending_class' that cut spending score into the 4 quartiles and label the new field by [q1, q2, q3, q4]. 
    split on target of 'spending_score'
    scale age and annual income. 
    '''
    
    train, validate, test = train_validate_test_split(df, target='zestimateerror', seed=123)
    train, validate, test = scale_my_data(train, validate, test)
    return df, train, validate, test