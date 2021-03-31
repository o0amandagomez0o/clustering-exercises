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





def get_mall():
    '''
    This function reads in the `mall_customers` datatset from the Codeup SQL DB
    '''
    
    sql = """
    select *
    from customers
    """
    return pd.read_sql(sql, get_connection('mall_customers'))




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



"""10722858, 10732347, 10739478, 10744507, 10753427, 10753829, 10777937, 10779619, 10792511, 10808647, 10811539, 10833654, 10852812, 10857130, 10858360, 10871677, 10879060, 10903921, 10929090, 10946379, 10956664, 10976131, 10979425, 10984080, 10984661, 11061050, 11065727, 11083106, 11106339, 11141309, 11187927, 11289757, 11289917, 11312124, 11367981, 11389003, 11391577, 11391972, 11393337, 11401519, 11420117, 11429175, 11433174, 11446756, 11451345, 11460552, 11460921, 11491470, 11496770, 11499166, 11499751, 11501340, 11501341, 11501342, 11552513, 11577176, 11594130, 11603473, 11605789, 11627049, 11637029, 11658743, 11694397, 11696784, 11705026, 11711539, 11717962, 11721753, 11733550, 11739891, 11743374, 11797465, 11828977, 11830465, 11917650, 11921077, 11923149, 11938901, 11957553, 11958628, 11961462, 11967869, 11969146, 11991059, 11999890, 12002715, 12027770, 12035176, 12035592, 12048224, 12057023, 12068159, 12099888, 12102046, 12114701, 12118682, 12121210, 12136147, 12137395, 12178305, 12196319, 12213076, 12224279, 12285822, 12347492, 12373769, 12383085, 12385712, 12402398, 12443331, 12454426, 12478591, 12492881, 12505219, 12519794, 12532988, 12535098, 12537644, 12541155, 12575721, 12607366, 12612211, 12613390, 12621730, 12641353, 12693966, 12749741, 12811794, 12814323, 12827519, 12847318, 12870253, 12892594, 12941764, 12955531, 12982361, 13020886, 13066981, 13067305, 13067643, 13071085, 13075560, 13083743, 13863275, 13880422, 13885693, 13893511, 13921492, 13960284, 13973642, 14008322, 14010551, 14012730, 14050918, 14074415, 14079874, 14088988, 14092694, 14097534, 14236060, 14254548, 14257065, 14269464, 14365030, 14430658, 14448410, 14455319, 14532131, 14604480, 14606311, 14619445, 14621355, 14626467, 14634203, 14637110, 14648524, 14655760, 14656105, 14671963, 14718350, 14734602, 17086759, 17098564, 17136356, 17139939, 17165634, 17165666, 17190654, 17193966, 17225336, 17243116, 17251843, 17280166, 17282392, 17295416, 162960529"""




