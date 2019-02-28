import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score,  GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin


housing = pd.read_csv('datasets/house-prices-advanced/train.csv')
test = pd.read_csv('datasets/house-prices-advanced/test.csv')

housing.set_index('Id', inplace = True)    
housing_labels = housing.SalePrice

year_attribs = ['YearsBeingBlt','YearsAddRemod','YearsGarageBlt']
neigh = housing.SalePrice.groupby(housing.Neighborhood).mean().sort_values().index.tolist()
def trans(housing):     

	global year_attribs

	housing['YearsBeingBlt'] = housing.YearBuilt - housing.YrSold
	housing['YearsAddRemod'] = housing.YearRemodAdd - housing.YrSold
	housing['YearsGarageBlt'] = housing.GarageYrBlt - housing.YrSold

	global neigh
	neigh_class = [0,0,0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8]
	neighs = dict(zip(neigh, neigh_class)) 

	housing.Neighborhood.replace(neighs, inplace = True)
	housing.Utilities.replace({'AllPub' : 3, 'NoSewr' : 2, 'NoSeWa': 1, 'ELO' : 0}, inplace = True)
	housing.ExterQual.replace({'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace = True)
	housing.ExterCond.replace({'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace = True)
	housing.BsmtQual.replace({'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1}, inplace = True)
	housing.BsmtCond.replace({'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1}, inplace = True)
	housing.BsmtExposure.replace({'Gd' : 3, 'Av' : 2, 'Mn' : 1, 'No' : 0}, inplace = True)
	housing.BsmtFinType1.replace({'GLQ' : 5, 'ALQ' : 4, 'BLQ':3, 'Rec' : 2, 'LwQ' : 1, 'Unf' : .1}, inplace = True)
	housing.BsmtFinType2.replace({'GLQ' : 5, 'ALQ' : 4, 'BLQ':3, 'Rec' : 2, 'LwQ' : 1, 'Unf' : .1}, inplace = True)
	housing.HeatingQC.replace({'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace = True)
	housing.CentralAir.replace({'Y' : 1, 'N' : 0}, inplace = True)
	housing.KitchenQual.replace({'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace = True)
	housing.Functional.replace({'Typ' : 6, 'Min1' : 5, 'Min2' : 4, 'Mod' : 3, 'Maj1' : 2, 
		'Maj2' : 1, 'Sev' : 0, 'Sal' : -1}, inplace = True)
	housing.FireplaceQu.replace({'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1}, inplace = True)
	housing.GarageFinish.replace({'Fin' : 3, 'RFn' : 2, 'Unf' : 1}, inplace = True)
	housing.GarageQual.replace({'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1}, inplace = True)
	housing.GarageCond.replace({'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1}, inplace = True)
	housing.PavedDrive.replace({'Y' : 2,  'P' : 1, 'N' : 0}, inplace = True)
	housing.PoolQC.replace({'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1}, inplace = True)
	housing.Fence.replace({'GdPrv' : 2, 'MnPrv' : 1, 'GdWo' : 2, 'MnWw' : 1}, inplace = True)


	Unused = ['TotalBsmtSF','YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']

	housing.drop(Unused, axis = 1, inplace = True) 

trans(housing)
trans(test)

housing.drop('SalePrice', axis = 1, inplace = True) 

cat_attribs = ['MSSubClass', 'MSZoning', 'Street','Alley','LotShape','LandContour',
'LotConfig','LandSlope','Condition1','Condition2','BldgType','HouseStyle',
'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
'Foundation','Heating','Electrical','GarageType','MiscFeature','SaleType','SaleCondition'


]
num_attribs = ['LotFrontage', 'LotArea','Utilities','OverallQual','OverallCond','MasVnrArea',
'ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','HeatingQC','CentralAir','1stFlrSF',
'2ndFlrSF','LowQualFinSF','GrLivArea','Neighborhood',
'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
'KitchenQual','TotRmsAbvGrd','Functional',
'Fireplaces','FireplaceQu','GarageFinish','GarageCars','GarageArea','GarageQual',
'GarageCond','PavedDrive',
'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',


 'PoolQC','Fence','MiscVal'
]

num_attribs += year_attribs

from scipy.stats import skew

skewed = housing[num_attribs].apply(lambda x : skew(x.dropna().astype(float))) 
skewed = skewed[skewed > .75].index


housing[skewed] = np.log1p(housing[skewed].astype('float32'))
test[skewed] = np.log1p(test[skewed].astype('float32'))    



def diff(a): 
    return set(housing[a].unique().tolist()) - set( test[a].unique().tolist())

from collections import defaultdict 
missing = defaultdict(list)
for cat in cat_attribs: 
     if housing[cat].unique().tolist() != test[cat].unique().tolist(): 
         missing[cat].extend(list(diff(cat))) 

for cat in missing:
	if len(missing[cat]) > 0:
		housing[cat].replace(missing[cat], np.nan, inplace = True)




cat_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy = 'most_frequent')),
	('1hot', OneHotEncoder()),
	])

num_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),
	('scaler', StandardScaler()),
	])


full_pipeline = ColumnTransformer([
	('num', num_pipeline, num_attribs),
	('cat', cat_pipeline, cat_attribs),
	])

housing_prepared = full_pipeline.fit_transform(housing) 



def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from xgboost import XGBRegressor
from scipy.stats import randint, uniform

xgb_reg = XGBRegressor()

	param_distribs = {
	    
		'n_estimators':randint(low = 100, high = 1000),
		'max_depth':randint(low = 1, high = 4),
		'colsample_bytree' : uniform(loc = 0, scale = .5),
		'min_child_weight' : uniform(loc = 0.5, scale = 1.5)


}

rnd_search = RandomizedSearchCV(xgb_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42,
                                verbose = 1, n_jobs = -1)


rnd_search.fit(housing_prepared, housing_labels)
model = rnd_search.best_estimator_


test.at[1358, 'MSSubClass'] = 20    #1358   
test_prepared = full_pipeline.transform(test)

pred = model.predict(test_prepared)
result = pd.Series(pred, index = test.Id)
result = pd.DataFrame(result, columns = ['SalePrice'])


