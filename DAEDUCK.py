import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

data = pd.read_csv("DATA.csv", encoding="CP949")
data = data.drop(["ID"], axis=1)
print(list(data.columns))
print()

data = pd.concat([data, pd.get_dummies(data.loc[:, ["Type", "Strip"]], columns=["Type", "Strip"])], axis=1)
data["Price"] = np.log(data["Price"])
print(list(data.columns))
print()

data = data[data.Vendor != "MTG"]
#data = data[data.Vendor != "DAEDUCK"]
data = data[data.Vendor != "KCC"]
data = data[data.Vendor != "FASTPRINT"]
data = data[data.Vendor != "SCC"]
data = data[data.Vendor != "SIMMTECH"]
data = data[data.Vendor != "ZDT"]
data = data[data.Vendor != "KINSUS"]
data = data[data.Vendor != "HDS"]
data = data[data.Vendor != "ASEM"]

cols = ['Year', 'PKG_Size', 'Sub_Array', 'Sub_T', 'Material_T', 'VIA_Land', 'VIA_Count',
'BF_Space', 'BF_Ratio_Space', 'T14_Layer4', 'MSAP', 'Etch_Back']

df = data.loc[:, cols + ["Price"]]
reg = sm.OLS(df["Price"], sm.add_constant(df.ix[:, :-1])).fit()
print(reg.summary())
print()

fig, ax = plt.subplots()
colormap = {"BOC":"Red", "EMMC":"Gold", "FBGA":"Gray", "LGA":"Green", "Module":"LightGreen", "Sensor":"YellowGreen", "UFD":"Violet", "UFS":"MediumBlue"}
#colormap = {"MTG":"Red", "DAEDUCK":"White", "KCC":"White", "FASTPRINT":"White", "SCC":"White", "SIMMTECH":"White", "ZDT":"White", "KINSUS":"White", "HDS":"White", "ASEM":"White"}
ax.scatter(x=df["Price"], y=reg.predict(sm.add_constant(df.ix[:, :-1])), c=[colormap[t] for t in data["Type"]], s=9)
ax.set(xlim=(-4.5, -0.5), ylim=(-4.5, -0.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), linewidth=1, color="Red", linestyle="--")
plt.xlabel("Actual Price", fontsize=10)
plt.ylabel("Predicted Price", fontsize=10)
plt.title(list(df.columns)[:-1], fontsize=11)
plt.show()

##[colormap[t] for t in data["Vendor"]]
x = pd.DataFrame(reg.params)
x.to_csv("DAEDUCK.csv")
