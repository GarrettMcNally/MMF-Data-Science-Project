import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor

def ordinal_cols(data):
    #create ordinal column for cut
    cutmap = {"Fair": 0,"Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    data["ord_cut"] = data["cut"].replace(cutmap)
    data.drop(columns=["cut"], inplace=True)

    #create oridinal column for color
    # train = pd.get_dummies(train,prefix = ["color"], columns = ["color"], drop_first=True)
    colormap = {"M": 0, "L": 1, "K": 2, "J": 3, "I": 4, "H": 5, "G": 6, "F": 7, "E": 8, "D": 9}
    data["ord_colour"] = data["color"].replace(colormap)
    data.drop(columns=["color"], inplace=True)

    #create oridinal column for clarity
    clarmap = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
    data["ord_clar"] = data["clarity"].replace(clarmap)
    data.drop(columns=["clarity"], inplace=True)

    return data

rawtrain = pd.read_csv("C:/Users/ruxer/Documents/python/MMF Data Science/train.csv", sep=",",  header=0, index_col=0)

ordtrain = ordinal_cols(rawtrain)

#assess data
# print(ordtrain.describe())


#plot the variables vs the price
# for col in ordtrain:
#     plt.scatter(ordtrain["price"], ordtrain[col])
#     plt.xlabel("Price")
#     plt.ylabel(col)
#     plt.show()

def remove_outliers(data):
    cleaned = data.copy()
    
    #remove outliters from these two from the graghs
    outliercols = ["table", "depth"]
    for col in outliercols:
        cleaned = cleaned[(cleaned[col] < 75) & (cleaned[col] > 45)]

    #get rid of bad data and bound these values based on table
    baddata = ["x","y","z"]
    for col in baddata:
        cleaned = cleaned.drop(cleaned[cleaned[col] == 0].index)
        cleaned = cleaned[cleaned[col] < 45]

    return cleaned

train = remove_outliers(ordtrain)

#seperate data into X and Y
Y_train = train["price"]
X_train = train.drop(columns=["price"])

#base model
model = sm.OLS(endog = Y_train, exog = sm.add_constant(X_train, prepend=False) ).fit()
# print(model.summary())

print(np.sqrt(np.average(model.resid**2)))

xgmodel = XGBRegressor()
xgmodel.fit(X = X_train, y = Y_train)

XGB_y = xgmodel.predict(X_train)
print(np.sqrt(mean_squared_error(Y_train, XGB_y)))


#########################################################################
##############            Testing Dataset                  ##############           
#########################################################################

rawtest = pd.read_csv("C:/Users/ruxer/Documents/python/MMF Data Science/test.csv", sep=",",  header=0, index_col=0)
test = ordinal_cols(rawtest)


xg_pred = xgmodel.predict(test)
print(xg_pred)
submission = pd.Series(xg_pred, name="price")
# submission.to_csv(path_or_buf="C:/Users/ruxer/Documents/python/MMF Data Science/submission.csv", sep = ",", header = 1)