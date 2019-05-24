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
 {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>amount_tsh</th>\n",
       "      <th>date_recorded</th>\n",
       "      <th>funder</th>\n",
       "      <th>gps_height</th>\n",
       "      <th>installer</th>\n",
       "      <th>longitude</th>\n",
       "      <th>latitude</th>\n",
       "      <th>wpt_name</th>\n",
       "      <th>num_private</th>\n",
       "      <th>basin</th>\n",
       "      <th>subvillage</th>\n",
       "      <th>region</th>\n",
       "      <th>region_code</th>\n",
       "      <th>district_code</th>\n",
       "      <th>lga</th>\n",
       "      <th>ward</th>\n",
       "      <th>population</th>\n",
       "      <th>public_meeting</th>\n",
       "      <th>recorded_by</th>\n",
       "      <th>scheme_management</th>\n",
       "      <th>scheme_name</th>\n",
       "      <th>permit</th>\n",
       "      <th>construction_year</th>\n",
       "      <th>extraction_type</th>\n",
       "      <th>extraction_type_group</th>\n",
       "      <th>extraction_type_class</th>\n",
       "      <th>management</th>\n",
       "      <th>management_group</th>\n",
       "      <th>payment</th>\n",
       "      <th>payment_type</th>\n",
       "      <th>water_quality</th>\n",
       "      <th>quality_group</th>\n",
       "      <th>quantity</th>\n",
       "      <th>quantity_group</th>\n",
       "      <th>source</th>\n",
       "      <th>source_type</th>\n",
       "      <th>source_class</th>\n",
       "      <th>waterpoint_type</th>\n",
       "      <th>waterpoint_type_group</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>69572</td>\n",
       "      <td>6000.0</td>\n",
       "      <td>2011-03-14</td>\n",
       "      <td>Roman</td>\n",
       "      <td>1390</td>\n",
       "      <td>Roman</td>\n",
       "      <td>34.938093</td>\n",
       "      <td>-9.856322</td>\n",
       "      <td>none</td>\n",
       "      <td>0</td>\n",
       "      <td>Lake Nyasa</td>\n",
       "      <td>Mnyusi B</td>\n",
       "      <td>Iringa</td>\n",
       "      <td>11</td>\n",
       "      <td>5</td>\n",
       "      <td>Ludewa</td>\n",
       "      <td>Mundindi</td>\n",
       "      <td>109</td>\n",
       "      <td>True</td>\n",
       "      <td>GeoData Consultants Ltd</td>\n",
       "      <td>VWC</td>\n",
       "      <td>Roman</td>\n",
       "      <td>False</td>\n",
       "      <td>1999</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>vwc</td>\n",
       "      <td>user-group</td>\n",
       "      <td>pay annually</td>\n",
       "      <td>annually</td>\n",
       "      <td>soft</td>\n",
       "      <td>good</td>\n",
       "      <td>enough</td>\n",
       "      <td>enough</td>\n",
       "      <td>spring</td>\n",
       "      <td>spring</td>\n",
       "      <td>groundwater</td>\n",
       "      <td>communal standpipe</td>\n",
       "      <td>communal standpipe</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>8776</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2013-03-06</td>\n",
       "      <td>Grumeti</td>\n",
       "      <td>1399</td>\n",
       "      <td>GRUMETI</td>\n",
       "      <td>34.698766</td>\n",
       "      <td>-2.147466</td>\n",
       "      <td>Zahanati</td>\n",
       "      <td>0</td>\n",
       "      <td>Lake Victoria</td>\n",
       "      <td>Nyamara</td>\n",
       "      <td>Mara</td>\n",
       "      <td>20</td>\n",
       "      <td>2</td>\n",
       "      <td>Serengeti</td>\n",
       "      <td>Natta</td>\n",
       "      <td>280</td>\n",
       "      <td>NaN</td>\n",
       "      <td>GeoData Consultants Ltd</td>\n",
       "      <td>Other</td>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "      <td>2010</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>wug</td>\n",
       "      <td>user-group</td>\n",
       "      <td>never pay</td>\n",
       "      <td>never pay</td>\n",
       "      <td>soft</td>\n",
       "      <td>good</td>\n",
       "      <td>insufficient</td>\n",
       "      <td>insufficient</td>\n",
       "      <td>rainwater harvesting</td>\n",
       "      <td>rainwater harvesting</td>\n",
       "      <td>surface</td>\n",
       "      <td>communal standpipe</td>\n",
       "      <td>communal standpipe</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>34310</td>\n",
       "      <td>25.0</td>\n",
       "      <td>2013-02-25</td>\n",
       "      <td>Lottery Club</td>\n",
       "      <td>686</td>\n",
       "      <td>World vision</td>\n",
       "      <td>37.460664</td>\n",
       "      <td>-3.821329</td>\n",
       "      <td>Kwa Mahundi</td>\n",
       "      <td>0</td>\n",
       "      <td>Pangani</td>\n",
       "      <td>Majengo</td>\n",
       "      <td>Manyara</td>\n",
       "      <td>21</td>\n",
       "      <td>4</td>\n",
       "      <td>Simanjiro</td>\n",
       "      <td>Ngorika</td>\n",
       "      <td>250</td>\n",
       "      <td>True</td>\n",
       "      <td>GeoData Consultants Ltd</td>\n",
       "      <td>VWC</td>\n",
       "      <td>Nyumba ya mungu pipe scheme</td>\n",
       "      <td>True</td>\n",
       "      <td>2009</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>vwc</td>\n",
       "      <td>user-group</td>\n",
       "      <td>pay per bucket</td>\n",
       "      <td>per bucket</td>\n",
       "      <td>soft</td>\n",
       "      <td>good</td>\n",
       "      <td>enough</td>\n",
       "      <td>enough</td>\n",
       "      <td>dam</td>\n",
       "      <td>dam</td>\n",
       "      <td>surface</td>\n",
       "      <td>communal standpipe multiple</td>\n",
       "      <td>communal standpipe</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>67743</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2013-01-28</td>\n",
       "      <td>Unicef</td>\n",
       "      <td>263</td>\n",
       "      <td>UNICEF</td>\n",
       "      <td>38.486161</td>\n",
       "      <td>-11.155298</td>\n",
       "      <td>Zahanati Ya Nanyumbu</td>\n",
       "      <td>0</td>\n",
       "      <td>Ruvuma / Southern Coast</td>\n",
       "      <td>Mahakamani</td>\n",
       "      <td>Mtwara</td>\n",
       "      <td>90</td>\n",
       "      <td>63</td>\n",
       "      <td>Nanyumbu</td>\n",
       "      <td>Nanyumbu</td>\n",
       "      <td>58</td>\n",
       "      <td>True</td>\n",
       "      <td>GeoData Consultants Ltd</td>\n",
       "      <td>VWC</td>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "      <td>1986</td>\n",
       "      <td>submersible</td>\n",
       "      <td>submersible</td>\n",
       "      <td>submersible</td>\n",
       "      <td>vwc</td>\n",
       "      <td>user-group</td>\n",
       "      <td>never pay</td>\n",
       "      <td>never pay</td>\n",
       "      <td>soft</td>\n",
       "      <td>good</td>\n",
       "      <td>dry</td>\n",
       "      <td>dry</td>\n",
       "      <td>machine dbh</td>\n",
       "      <td>borehole</td>\n",
       "      <td>groundwater</td>\n",
       "      <td>communal standpipe multiple</td>\n",
       "      <td>communal standpipe</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>19728</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2011-07-13</td>\n",
       "      <td>Action In A</td>\n",
       "      <td>0</td>\n",
       "      <td>Artisan</td>\n",
       "      <td>31.130847</td>\n",
       "      <td>-1.825359</td>\n",
       "      <td>Shuleni</td>\n",
       "      <td>0</td>\n",
       "      <td>Lake Victoria</td>\n",
       "      <td>Kyanyamisa</td>\n",
       "      <td>Kagera</td>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>Karagwe</td>\n",
       "      <td>Nyakasimbi</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>GeoData Consultants Ltd</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>gravity</td>\n",
       "      <td>other</td>\n",
       "      <td>other</td>\n",
       "      <td>never pay</td>\n",
       "      <td>never pay</td>\n",
       "      <td>soft</td>\n",
       "      <td>good</td>\n",
       "      <td>seasonal</td>\n",
       "      <td>seasonal</td>\n",
       "      <td>rainwater harvesting</td>\n",
       "      <td>rainwater harvesting</td>\n",
       "      <td>surface</td>\n",
       "      <td>communal standpipe</td>\n",
       "      <td>communal standpipe</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  amount_tsh date_recorded        funder  gps_height     installer  \\\n",
       "0  69572      6000.0    2011-03-14         Roman        1390         Roman   \n",
       "1   8776         0.0    2013-03-06       Grumeti        1399       GRUMETI   \n",
       "2  34310        25.0    2013-02-25  Lottery Club         686  World vision   \n",
       "3  67743         0.0    2013-01-28        Unicef         263        UNICEF   \n",
       "4  19728         0.0    2011-07-13   Action In A           0       Artisan   \n",
       "\n",
       "   longitude   latitude              wpt_name  num_private  \\\n",
       "0  34.938093  -9.856322                  none            0   \n",
       "1  34.698766  -2.147466              Zahanati            0   \n",
       "2  37.460664  -3.821329           Kwa Mahundi            0   \n",
       "3  38.486161 -11.155298  Zahanati Ya Nanyumbu            0   \n",
       "4  31.130847  -1.825359               Shuleni            0   \n",
       "\n",
       "                     basin  subvillage   region  region_code  district_code  \\\n",
       "0               Lake Nyasa    Mnyusi B   Iringa           11              5   \n",
       "1            Lake Victoria     Nyamara     Mara           20              2   \n",
       "2                  Pangani     Majengo  Manyara           21              4   \n",
       "3  Ruvuma / Southern Coast  Mahakamani   Mtwara           90             63   \n",
       "4            Lake Victoria  Kyanyamisa   Kagera           18              1   \n",
       "\n",
       "         lga        ward  population public_meeting              recorded_by  \\\n",
       "0     Ludewa    Mundindi         109           True  GeoData Consultants Ltd   \n",
       "1  Serengeti       Natta         280            NaN  GeoData Consultants Ltd   \n",
       "2  Simanjiro     Ngorika         250           True  GeoData Consultants Ltd   \n",
       "3   Nanyumbu    Nanyumbu          58           True  GeoData Consultants Ltd   \n",
       "4    Karagwe  Nyakasimbi           0           True  GeoData Consultants Ltd   \n",
       "\n",
       "  scheme_management                  scheme_name permit  construction_year  \\\n",
       "0               VWC                        Roman  False               1999   \n",
       "1             Other                          NaN   True               2010   \n",
       "2               VWC  Nyumba ya mungu pipe scheme   True               2009   \n",
       "3               VWC                          NaN   True               1986   \n",
       "4               NaN                          NaN   True                  0   \n",
       "\n",
       "  extraction_type extraction_type_group extraction_type_class management  \\\n",
       "0         gravity               gravity               gravity        vwc   \n",
       "1         gravity               gravity               gravity        wug   \n",
       "2         gravity               gravity               gravity        vwc   \n",
       "3     submersible           submersible           submersible        vwc   \n",
       "4         gravity               gravity               gravity      other   \n",
       "\n",
       "  management_group         payment payment_type water_quality quality_group  \\\n",
       "0       user-group    pay annually     annually          soft          good   \n",
       "1       user-group       never pay    never pay          soft          good   \n",
       "2       user-group  pay per bucket   per bucket          soft          good   \n",
       "3       user-group       never pay    never pay          soft          good   \n",
       "4            other       never pay    never pay          soft          good   \n",
       "\n",
       "       quantity quantity_group                source           source_type  \\\n",
       "0        enough         enough                spring                spring   \n",
       "1  insufficient   insufficient  rainwater harvesting  rainwater harvesting   \n",
       "2        enough         enough                   dam                   dam   \n",
       "3           dry            dry           machine dbh              borehole   \n",
       "4      seasonal       seasonal  rainwater harvesting  rainwater harvesting   \n",
       "\n",
       "  source_class              waterpoint_type waterpoint_type_group  \n",
       "0  groundwater           communal standpipe    communal standpipe  \n",
       "1      surface           communal standpipe    communal standpipe  \n",
       "2      surface  communal standpipe multiple    communal standpipe  \n",
       "3  groundwater  communal standpipe multiple    communal standpipe  \n",
       "4      surface           communal standpipe    communal standpipe  "
      ]
     }
      
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
