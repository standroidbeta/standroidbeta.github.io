---
layout: post
title: Predictive Modeling Kaggle Challenge
subtitle: Tanzanian Water Pump Functionality Predictions
bigimg: /img/tanzania-flag.jpg
tags: [Kaggle, Predictive Modeling, Tanzanian Water Pumps]
---

## Purpose
This is my second data science project at Lambda School where I participated in a week-long private Kaggle Challenge to predict the functonality of water pumps in Tanzania. For the record, I did not place in the top half of my class but I was able to predict a respectible accuracy by using an XGBoost classification model using both randomized search cross-validation and grid search cross-validation techniques.

This was a really fun challenge because I was able to practically apply concepts that I have learned at Lambda School with real-world data that can be used for social good.

Continue reading if you dare to know more about my journey!

## Tanzanian Water Pump Kaggle Challenge Overview
![Tanzania Water Pump](/img/tanzania-water-pump.jpeg)

### Predict which water pumps are faulty.
Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

This predictive modeling challenge comes from DrivenData, an organization who helps non-profits by hosting data science competitions for social impact. The competition has open licensing: "The data is available for use outside of DrivenData." We are reusing the data on Kaggle's InClass platform so we can run a weeklong challenge just for your Lambda School Data Science cohort.

The data comes from the Taarifa waterpoints dashboard, which aggregates data from the Tanzania Ministry of Water. In their own words:

_Taarifa is an open source platform for the crowd sourced reporting and triaging of infrastructure related issues. Think of it as a bug tracker for the real world which helps to engage citizens with their local government. We are currently working on an Innovation Project in Tanzania, with various partners._

![Taarifa Dashboard](/img/taarifadashboard.png)

Go here to get the complete Kaggle challenge info.

[Tanzanian Water Pump Kaggle Challenge](https://www.kaggle.com/c/ds3-predictive-modeling-challenge/overview)

## Importing the Data and an Overloaded Toolbox

I started with retrieving the data from Kaggle like this:
```python
!kaggle competitions download -c ds3-predictive-modeling-challenge
```
I then imported all of the Python Library tools that I thought that I might need to use for this project. It turned out that I only used a few of them. But hey, better to be prepared for anything than for nothing.

```python

# Loading potential tools needed not all are used
%matplotlib inline
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from pdpbox.pdp import pdp_isolate, pdp_plot
from pdpbox.pdp import pdp_interact, pdp_interact_plot
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
```
### Examming the Dataset

The data came pre-split as test_features.csv.zip, train_features.csv.zip, and train_labels.csv.zip files.
Here is my quick examination of the data imported.

```python
train_features = pd.read_csv('/home/seek/Documents/GitHub/DS-Project-2---Predictive-Modeling-Challenge/train_features.csv.zip')
pd.options.display.max_columns = 40
train_features.head()
```
---
,id,amount_tsh,date_recorded,funder,gps_height,installer,longitude,latitude,wpt_name,num_private,basin,subvillage,region,region_code,district_code,lga,ward,population,public_meeting,recorded_by,scheme_management,scheme_name,permit,construction_year,extraction_type,extraction_type_group,extraction_type_class,management,management_group,payment,payment_type,water_quality,quality_group,quantity,quantity_group,source,source_type,source_class,waterpoint_type,waterpoint_type_group
0,69572,6000,2011-03-14,Roman,1390,Roman,34.938093,-9.856322,none,0,Lake Nyasa,Mnyusi B,Iringa,11,5,Ludewa,Mundindi,109,True,GeoData Consultants Ltd,VWC,Roman,False,1999,gravity,gravity,gravity,vwc,user-group,pay annually,annually,soft,good,enough,enough,spring,spring,groundwater,communal standpipe,communal standpipe
1,8776,0,2013-03-06,Grumeti,1399,GRUMETI,34.698766,-2.147466,Zahanati,0,Lake Victoria,Nyamara,Mara,20,2,Serengeti,Natta,280,NaN,GeoData Consultants Ltd,Other,NaN,True,2010,gravity,gravity,gravity,wug,user-group,never pay,never pay,soft,good,insufficient,insufficient,rainwater harvesting,rainwater harvesting,surface,communal standpipe,communal standpipe
2,34310,25,2013-02-25,Lottery Club,686,World vision,37.460664,-3.821329,Kwa Mahundi,0,Pangani,Majengo,Manyara,21,4,Simanjiro,Ngorika,250,True,GeoData Consultants Ltd,VWC,Nyumba ya mungu pipe scheme,True,2009,gravity,gravity,gravity,vwc,user-group,pay per bucket,per bucket,soft,good,enough,enough,dam,dam,surface,communal standpipe multiple,communal standpipe
3,67743,0,2013-01-28,Unicef,263,UNICEF,38.486161,-11.155298,Zahanati Ya Nanyumbu,0,Ruvuma / Southern Coast,Mahakamani,Mtwara,90,63,Nanyumbu,Nanyumbu,58,True,GeoData Consultants Ltd,VWC,NaN,True,1986,submersible,submersible,submersible,vwc,user-group,never pay,never pay,soft,good,dry,dry,machine dbh,borehole,groundwater,communal standpipe multiple,communal standpipe
4,19728,0,2011-07-13,Action In A,0,Artisan,31.130847,-1.825359,Shuleni,0,Lake Victoria,Kyanyamisa,Kagera,18,1,Karagwe,Nyakasimbi,0,True,GeoData Consultants Ltd,NaN,NaN,True,0,gravity,gravity,gravity,other,other,never pay,never pay,soft,good,seasonal,seasonal,rainwater harvesting,rainwater harvesting,surface,communal standpipe,communal standpipe
---
      
## First Baseline Model Prediction



### XGBoost Classification with RandomizedSearchCV.

## Second Baseline Model Prediction

### XGBoost Classification on Test Train Split Dataset with RandomizedSearchCV.

### A Fun Little Confusion Matrix on the Prediction Model for Clarification.

## Third Cleaned Data Model Prediction.

### Overdone Cleaning Funcion.

### XGBoost Classification on Test Train Split Dataset with GridSearchCV and adjusted Hyper-parameters.

### Hey! Here is Another Model Confusion Matrix for Some More Clarification.

## Here Are Some Fun but Simple Visualizations for Your Pleasure.

## Conclusion
