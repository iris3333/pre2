import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("PCB.csv", encoding="CP949")
data = data.drop(["ID"], axis=1)
print(list(data.columns))
print()

data = pd.concat([data, pd.get_dummies(data.loc[:, ["Type", "Strip", "Vendor"]], columns=["Type", "Strip", "Vendor"])], axis=1)
data["Price"] = np.log(data["Price"])
print(list(data.columns))
print()

data = data.drop([35, 60, 108, 118])

cols = ['Year', 'PKG_Size', 'Ball_Count', 'Sub_Array', 'Sub_T', 'Material_T', 'VIA_Drill', 'VIA_Land', 'VIA_Count',
'BF_Pitch', 'BF_Width', 'BF_Space', 'BF_Ratio_Width', 'BF_Ratio_Space',
'TR_Pitch', 'TR_Width', 'TR_Space', 'TR_Ratio_Width', 'TR_Ratio_Space',
'Layer4', 'T14_Layer4', 'Mold_Large', 'MSAP', 'VOP', 'Etch_Back', 'Hard_Au',
'Type_BOC', 'Type_EMMC', 'Type_FBGA', 'Type_LGA', 'Type_Module', 'Type_Sensor', 'Type_UFD', 'Type_UFS',
'Strip_A', 'Strip_B', 'Strip_C', 'Strip_D',
'Vendor_DAEDUCK', 'Vendor_FASTPRINT', 'Vendor_HDS', 'Vendor_KCC', 'Vendor_KINSUS', 'Vendor_MTG', 'Vendor_SCC', 'Vendor_SIMMTECH', 'Vendor_ZDT']

df = data.loc[:, ["VIA_Count", "PKG_Size", "Layer4"] + ["Price"]]

test = []; count = 0
for rand in range(0, 100):
	X_train, X_test, y_train, y_test = train_test_split(df.ix[:, :-1], df["Price"], test_size=0.2, random_state=rand)
	reg_tree = DecisionTreeRegressor(min_samples_leaf=3).fit(X_train, y_train)
	score = round(reg_tree.score(X_test, y_test), 3)
	test.append(score)
	if score < 0.6:
		count += 1

print(count)
print(pd.DataFrame(test).describe())
print()

##print([(data.columns[i], round(float(value), 3)) for i, value in enumerate(reg_tree.feature_importances_)])

X_train, X_test, y_train, y_test = train_test_split(df.ix[:, :-1], df["Price"], test_size=0.2, random_state=0)
reg_tree = DecisionTreeRegressor(min_samples_leaf=4).fit(X_train, y_train)

fig, ax = plt.subplots()
ax.scatter(x=y_train, y=reg_tree.predict(X_train), c="Black", s=4)
ax.scatter(x=y_test, y=reg_tree.predict(X_test), c="Red", s=4)
ax.set(xlim=(-4.5, -0.5), ylim=(-4.5, -0.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), color="Red", linewidth=1, linestyle="--")
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title(list(df.columns)[:-1])
plt.show()

