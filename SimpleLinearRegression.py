from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("lsd.dat", delim_whitespace=True, names=["TissueConcentration","MathScore"])
model = LinearRegression()
model.fit(data[["TissueConcentration"]], data.MathScore)

#Results
slope = model.coef_
intercept = model.intercept_
x = data.TissueConcentration
# yHat = slope * x + intercept
yHat = model.predict(data[["TissueConcentration"]])
print("Intercept: ", intercept)
print("Slope/Coefficient : ", slope)
print("Prediction: ", yHat)

#Graph
plt.xlabel("Tissue Concentration")
plt.ylabel("Math Score")
plt.scatter(data.TissueConcentration, data.MathScore)
plt.plot(x, yHat)
plt.title("Simple Linear Regression")
plt.show()