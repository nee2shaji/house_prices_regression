import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.cross_validation import cross_val_score

##----------function for mean square 
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))    #squared mean error
    return(rmse)

##-----------calculate skewness of data
def calc_skewed_feats(numeric_feats):
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))#don't consider missing values for calculation purpose
    skewed_feats = skewed_feats[skewed_feats > 0.7] #threshold of skewness
    skewed_feats = skewed_feats.index
    return skewed_feats

##----------append non numeric data to end
def update_full_data(full_data, columns):
    non_numeric_cols = ['MSSubClass']
    dummy_data = []
    for col in columns:
        if full_data[col].dtype.name == 'object' or col in non_numeric_cols:
            dummy_data.append(pd.get_dummies(full_data[col].values.astype(str), col))###for non numeric data append column to end and convert it into dummies
            dummy_data[-1].index = full_data.index
            del full_data[col]
    full_data = pd.concat([full_data] + dummy_data, axis=1)
    full_data = full_data.fillna(full_data.mean())
    return full_data

train = pd.read_csv("train_or.csv")
test = pd.read_csv("test.csv")
full_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],   #concatenate training data+testing data
                      test.loc[:,'MSSubClass':'SaleCondition']))

train["SalePrice"] = np.log1p(train["SalePrice"]) #log transform the target:
numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index  #log transform skewed numeric features:
skewed_feats = calc_skewed_feats(numeric_feats) #calculate skewness of data
full_data[skewed_feats] = np.log1p(full_data[skewed_feats]) #normalization applied on skewed data

columns = full_data.columns.values  # Label nominal variables to numbers
full_data = update_full_data(full_data, columns)

##-------------separate training data and testing data
X_train = full_data[:train.shape[0]] 
X_test = full_data[train.shape[0]:]
y = train.SalePrice

##------------Applying Lasso model
m_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(m_lasso).mean()     ##root mean squared error

##------------Applying Xgboost model
m_xgb = xgb.XGBRegressor(n_estimators=10000, max_depth=5,min_child_weight=1.5,reg_alpha=0.75,reg_lambda=0.45,learning_rate=0.07,subsample=0.95)
m_xgb.fit(X_train, y)   ##fit gradient boosting classifier

##------------The Numpy exp function np.expm1()----->calulates exp(x)-1 for all the elements in the array.
p_xgb = np.expm1(m_xgb.predict(X_test))
p_lasso = np.expm1(m_lasso.predict(X_test))

##-----------Give weights to calculated predicted value
preds = 0.75*p_lasso + 0.25*p_xgb
(pd.DataFrame({"id":test.Id, "SalePrice":preds})).to_csv("sur_sol.csv", index = False);
