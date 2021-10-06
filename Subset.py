import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

data = pd.read_csv("PCB.csv", encoding="CP949")
data = data.drop(["ID", "Vendor"], axis=1)
print(list(data.columns))
print()

data = pd.concat([data, pd.get_dummies(data.loc[:, ["Type", "Strip"]], columns=["Type", "Strip"])], axis=1)
data["Price"] = np.log(data["Price"])
print(list(data.columns))
print()

cols = ["PKG_Size", "Ball_Count", "Sub_Array", "Sub_T", "Material_T", "VIA_Drill", "VIA_Land", "VIA_Count",
"BF_Pitch", "BF_Width", "BF_Space", "BF_Ratio_Width", "BF_Ratio_Space",
"TR_Pitch", "TR_Width", "TR_Space", "TR_Ratio_Width", "TR_Ratio_Space",
"Type_BOC", "Type_EMMC", "Type_FBGA", "Type_LGA", "Type_Module", "Type_Sensor", "Type_UFD",  "Type_UFS",
"Strip_A", "Strip_B", "Strip_C", "Strip_D", "Year", "Layer4", "T14_Layer4", "Mold_Large", "MSAP", "VOP", "Etch_Back", "Hard_Au"]

ncol = 2
report = []

for col in list(itertools.combinations(cols, ncol)):
	df = data.loc[:, list(col) + ["Price"]]
	reg = sm.OLS(df["Price"], sm.add_constant(df.ix[:, :-1])).fit()
	count_p = 0
	for i in range(len(reg.pvalues)):
		if reg.pvalues[i] < 0.05:
			count_p += 0
		else:
			count_p += 1
	if count_p == 0:
		vif = [variance_inflation_factor(reg.model.exog, j) for j in range(reg.model.exog.shape[1])]
		count_vif = 0
		for j in range(reg.model.exog.shape[1]):
			if vif[j] < 10:
				count_vif += 0
			else:
				count_vif += 1
		if count_vif == 0:
			if jarque_bera(reg.resid)[1] > 0.05:
				if het_breuschpagan(reg.resid, reg.model.exog)[3] > 0.05:
					report.append((col, reg.rsquared_adj))

report = pd.DataFrame(report).rename(columns={0:"Subset", 1:"Adj_R2"})
report.to_csv("Subset.csv")

