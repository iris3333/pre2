import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm

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

df = data.loc[:, ["VIA_Count", "PKG_Size", "Layer4"] + ["Price"]]
reg = sm.OLS(df["Price"], sm.add_constant(df.ix[:, :-1])).fit()
print(reg.summary())
print()

test = pd.read_csv("PCB_TEST.csv", encoding="CP949")
test = test.loc[:, ["VIA_Count", "PKG_Size", "Layer4"] + ["Price"]]
test["Price"] = np.log(test["Price"])
test.index = range(len(test))

result = pd.DataFrame()
result["Log_Price"] = test["Price"]
result["Log_Predicted"] = reg.predict(sm.add_constant(test.ix[:, :-1]))
result["Price"] = np.e**result["Log_Price"]
result["Predicted"] = np.e**result["Log_Predicted"]
print(result)
print()

test_r2 = r2_score(y_true=result["Log_Price"], y_pred=result["Log_Predicted"])
print(test_r2)

test_adj_r2 = 1 - (1-test_r2)*(len(test)-1)/(len(test)-len(test.ix[:, :-1].columns)-1)
print(test_adj_r2)
print()

fig, ax = plt.subplots()
ax.scatter(x=df["Price"], y=reg.predict(sm.add_constant(df.ix[:, :-1])), c="Black", s=9)
ax.scatter(x=result["Log_Price"], y=result["Log_Predicted"], c="Red", s=9)
ax.set(xlim=(-4.5, -0.5), ylim=(-4.5, -0.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), linewidth=1, color="Red", linestyle="--")
plt.xlabel("Actual Price", fontsize=10)
plt.ylabel("Predicted Price", fontsize=10)
plt.title(list(df.columns)[:-1], fontsize=11)
plt.show()

