import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

data = pd.read_csv("PCB_v1.csv", encoding="CP949")
data = data[data.Select == 1].drop(["Year", "NO", "Select", "Type", "Vendor"], axis=1)
print("Columns\n{}".format(list(data.columns)))
print()

data = data[data.Type_FBGA == 1]
df = data.loc[:, ["PKG_Size_Area", "Layer2", "BF_Pitch", "Price"]]

reg_statsmodels = sm.OLS(df["Price"], sm.add_constant(df.ix[:, :-1])).fit()
print(reg_statsmodels.summary())
print()

fig, ax = plt.subplots()
ax.scatter(x=df["Price"], y=reg_statsmodels.predict(sm.add_constant(df.ix[:, :-1])), c="Black", s=4)
ax.set(xlim=(0, 0.36), ylim=(0, 0.36))
ax.plot(ax.get_xlim(), ax.get_ylim(), color="Red", linewidth=1, linestyle="--")
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title(list(df.columns)[:-1])
plt.show()

qq = sm.ProbPlot(reg_statsmodels.resid, dist=stats.norm, fit=True).qqplot(markerfacecolor="Black", markeredgecolor="Black", markersize=2)
sm.qqline(qq.axes[0], line="45", fmt="r--")
plt.title("QQ-plot of Residuals")
plt.show()

reg_lasso_cv = LassoCV(fit_intercept=True, normalize=False, alphas=(0.01, 0.1, 1, 10), cv=None).fit(df.ix[:, :-1], df.Price)
print("Lasso")
print("Alpha", reg_lasso_cv.alpha_)
print(dict(zip(df.ix[:, :-1].columns, reg_lasso_cv.coef_)), reg_lasso_cv.intercept_)
print("R2", round(reg_lasso_cv.score(df.ix[:, :-1], df.Price), 3))
print()

data["Predicted"] = reg_statsmodels.predict(sm.add_constant(df.ix[:, :-1]))
data["Residual"] = abs(data["Predicted"]-data["Price"])

