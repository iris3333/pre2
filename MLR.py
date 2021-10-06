import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
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

data = data.drop([35, 60, 108, 118])
data.index = range(len(data))
print(data.index)
print()

df = data.loc[:, ["VIA_Count", "PKG_Size", "Layer4"] + ["Price"]]
reg = sm.OLS(df["Price"], sm.add_constant(df.ix[:, :-1])).fit()
print(reg.summary())
print()

vif = pd.DataFrame()
if reg.model.exog.shape[1] == len(df.columns):
	vif["Feature"] = ["Intercept"] + list(df.columns)[:-1]
else:
	vif["Feature"] = list(df.columns)[:-1]
vif["VIF"] = [variance_inflation_factor(reg.model.exog, i) for i in range(reg.model.exog.shape[1])]
print(vif.round(3))
print()

print("Normality (Jarque-Bera P-value)", round(jarque_bera(reg.resid)[1], 3))
print("Homoscedasticity (Breusch-Pagan P-value)", round(het_breuschpagan(reg.resid, reg.model.exog)[3], 3))
print()

outlier = pd.DataFrame(reg.outlier_test(method="bonf", alpha=0.05))
outlier = outlier.rename(columns={"student_resid":"resid", "unadj_p":"unadj_p", "bonf(p)":"bonf_p"})
print(outlier[outlier.bonf_p < 0.05])
print()

leverage = OLSInfluence(reg).summary_frame().loc[:, ["hat_diag"]]
print(leverage[leverage.hat_diag > 0.2])
print()

influence = OLSInfluence(reg).summary_frame().loc[:, ["cooks_d"]]
print(influence[influence.cooks_d > (4/(len(df)-len(df.columns)-1))])
print()

fig, ax = plt.subplots()
ax.scatter(x=df["Price"], y=reg.predict(sm.add_constant(df.ix[:, :-1])), c="Black", s=9)
ax.set(xlim=(-4.5, -0.5), ylim=(-4.5, -0.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), linewidth=1, color="Red", linestyle="--")
plt.xlabel("Actual Price", fontsize=10)
plt.ylabel("Predicted Price", fontsize=10)
plt.title(list(df.columns)[:-1], fontsize=11)
plt.show()

fig, ax = plt.subplots()
qq = sm.ProbPlot(reg.resid, dist=stats.norm, fit=True)
qq = qq.qqplot(ax=ax, markerfacecolor="Black", markeredgecolor="Black", markersize=2)
ax.set(xlim=(-6, 6), ylim=(-6, 6))
sm.qqline(qq.axes[0], line="45", fmt="r--")
plt.xlabel("Theoretical Quantiles", fontsize=10)
plt.ylabel("Sample Quantiles", fontsize=10)
plt.title("Normality", fontsize=11)
plt.show()

fig, ax = plt.subplots()
ax.scatter(x=reg.predict(sm.add_constant(df.ix[:, :-1])), y=reg.resid, c="Black", s=9)
ax.set(xlim=(-4.5, -0.5), ylim=(-1.5, 1.5))
ax.plot(ax.get_xlim(), (0, 0), linewidth=1, color="Red", linestyle="--")
plt.xlabel("Fitted Values", fontsize=10)
plt.ylabel("Residuals", fontsize=10)
plt.title("Homoscedasticity", fontsize=11)
plt.show()

fig, ax = plt.subplots()
fig = sm.graphics.influence_plot(reg, ax=ax, criterion="cooks", alpha=0.01, plot_alpha=0.7)
ax.set(xlim=(0, 0.3), ylim=(-8, 8))
plt.xlabel("H Leverage", fontsize=10)
plt.ylabel("Studentized Residuals", fontsize=10)
plt.title("Influence (Circle Size for Cook's Distance)", fontsize=11)
plt.show()

