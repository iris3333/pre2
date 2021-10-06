import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("PCB.csv", encoding="CP949")
data = data.drop(["ID", "Vendor"], axis=1)
print("Columns\n{}".format(list(data.columns)))
print()

col = "Price"
print("Summary\n{}\n".format(data[col].describe()))
print("Value Counts (Unique {})\n{}".format(len(list(set(data[data[col].isnull() == False][col]))), data[col].value_counts()))
print()

def Categorical(df, col, y):
	freq = pd.crosstab(index=df[col], columns="count")
	freq.plot(kind="bar", rot=0, color="Gray", ylim=(0, y), legend=False)
	plt.xlabel("")
	plt.title(col)
	plt.show()

def Continuous(df, col, xmin, xmax):
	df[col].plot.hist(bins=np.linspace(xmin, xmax, 20), grid=False, color="Gray", cumulative=False)
	plt.xlabel("")
	plt.ylabel("Frequency")
	plt.title(col)
	plt.show()

Categorical(df=data, col=col, y=0)
Continuous(df=data, col=col, xmin=0, xmax=0)

two_freq = pd.crosstab(index=data[col], columns=data["Type"], margins=False)
print(two_freq)
print()

