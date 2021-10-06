import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

data = pd.read_csv("PCB.csv", encoding="CP949")
data = data.drop(["ID", "Vendor"], axis=1)
print(list(data.columns))
print()

data = pd.concat([data, pd.get_dummies(data.loc[:, ["Type", "Strip"]], columns=["Type", "Strip"])], axis=1)
data["Price"] = np.log(data["Price"])
print(list(data.columns))
print()

data = data.drop([35, 60, 108, 118])
data.index = range(len(data))
print(data.index)
print()

for rand in range(699, 700):

	X = data.loc[:, ["VIA_Count", "PKG_Size", "Layer4", "Type_EMMC"]]
	y = data.loc[:, ["Price"]]
	
	result = []
	kf = KFold(n_splits=10, random_state=rand, shuffle=True) ##148

	for train_index, test_index in kf.split(X):
		X_train, y_train, X_test, y_test = X.ix[train_index, :], y.ix[train_index, :], X.ix[test_index, :], y.ix[test_index, :]
		reg = sm.OLS(y_train, sm.add_constant(X_train)).fit()

		train_adj_r2 = reg.rsquared_adj
		test_r2 = r2_score(y_true=y_test, y_pred=reg.predict(sm.add_constant(X_test)))
		test_adj_r2 = 1 - (1-test_r2)*(len(X_test)-1)/(len(X_test)-len(X_test.columns)-1)
		print(test_r2)

		a = pd.DataFrame()
		a["Price"] = y_test["Price"]
		a["Predicted"] = reg.predict(sm.add_constant(X_test))
		a["SSR"] = a.Predicted - a.Price.mean()
		a["TSS"] = a.Price - a.Price.mean()
		print(sum(a.SSR**2)/sum(a.TSS**2))
		print()

		count_p = 0
		for i in range(len(reg.pvalues)):
			if reg.pvalues[i] < 0.05:
				count_p += 0
			else:
				count_p += 1
		if count_p == 0:
			x1 = 1
		else:
			x1 = None

		vif = [variance_inflation_factor(reg.model.exog, j) for j in range(reg.model.exog.shape[1])]
		count_vif = 0
		for j in range(reg.model.exog.shape[1]):
			if vif[j] < 10:
				count_vif += 0
			else:
				count_vif += 1
		if count_vif == 0:
			x2 = 1
		else:
			x2 = None

		if jarque_bera(reg.resid)[1] > 0.05:
			x3 = 1
		else:
			x3 = None

		if het_breuschpagan(reg.resid, reg.model.exog)[3] > 0.05:
			x4 = 1
		else:
			x4 = None

		result.append((train_adj_r2, test_adj_r2, x1, x2, x3, x4))

	result = pd.DataFrame(result).rename(columns={0:"Train", 1:"Test", 2:"P", 3:"VIF", 4:"JB", 5:"BP"})
	if sum(result["P"].isnull())==0 and sum(result["VIF"].isnull())==0 and sum(result["JB"].isnull())==0 and sum(result["BP"].isnull())==0:
		print("Random Number", rand)
		print(result)
		print()


