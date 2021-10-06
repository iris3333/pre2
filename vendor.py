import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm

data = pd.read_csv("PCB_v1.csv", encoding="CP949")
print("Columns\n{}".format(list(data.columns)))
print()

vendor = "DAEDUCK"

df = data[data.Vendor == vendor].drop(["Year", "NO", "Select", "Type", "Vendor"], axis=1)
print("Columns\n{}".format(list(df.columns)))
print()

reg_statsmodels = sm.OLS(df["Price"], sm.add_constant(df.ix[:, 0:-1])).fit()
print(reg_statsmodels.summary())
print()

fig, ax = plt.subplots()
ax.scatter(x=df["Price"], y=reg_statsmodels.predict(sm.add_constant(df.ix[:, 0:-1])), c="Black", s=4)
ax.set(xlim=(0, 0.4), ylim=(0, 0.4))
ax.plot(ax.get_xlim(), ax.get_ylim(), color="Red", linewidth=1, linestyle="--")
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title(vendor)
plt.show()

qq = sm.ProbPlot(reg_statsmodels.resid, dist=stats.norm, fit=True).qqplot(markerfacecolor="Black", markeredgecolor="Black", markersize=2)
sm.qqline(qq.axes[0], line="45", fmt="r--")
plt.title("QQ-plot of Residuals")
plt.show()

