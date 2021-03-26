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
    
select *
from properties_2017
join predictions_2017 using(parcelid)
limit 50;

select *
from properties_2017 as prop17
join predictions_2017 using(parcelid)
;

select *
from properties_2016 as prop16
join predictions_2016 using(parcelid)
;

select count(*)
from properties_2016 as prop16
join predictions_2016 using(parcelid)
left join properties_2017 as prop17 
on prop16.parcelid=prop17.parcelid;

select count(parcelid)
from properties_2016 as prop16
join predictions_2016 using(parcelid)
left join properties_2017 as prop17 using(parcelid)
left join predictions_2017 using(parcelid);

select count(parcelid)
from properties_2016 as prop16
join predictions_2016 using(parcelid)
join properties_2017 as prop17 using(parcelid)
left join predictions_2017 using(parcelid)
;

select *
from properties_2017 as prop17
join predictions_2017 using(parcelid)
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
;#77614rows

select count(*)
from unique_properties
;#2955825rows

select *
from properties_2017 as prop17
join predictions_2017 using(parcelid)
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
join unique_properties using(parcelid);

select *
from properties_2016 as prop16
join predictions_2016 using(parcelid)
left join airconditioningtype
ON prop16.airconditioningtypeid=airconditioningtype.airconditioningtypeid
left join architecturalstyletype
ON prop16.architecturalstyletypeid=architecturalstyletype.architecturalstyletypeid
left join buildingclasstype
ON prop16.buildingclasstypeid=buildingclasstype.buildingclasstypeid
left join heatingorsystemtype
ON prop16.heatingorsystemtypeid=heatingorsystemtype.heatingorsystemtypeid
left join propertylandusetype
ON prop16.propertylandusetypeid=propertylandusetype.propertylandusetypeid
left join storytype
ON prop16.storytypeid=storytype.storytypeid
left join typeconstructiontype
ON prop16.typeconstructiontypeid=typeconstructiontype.typeconstructiontypeid
join unique_properties using(parcelid);