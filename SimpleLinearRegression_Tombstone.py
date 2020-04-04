import pandas as pd
import matplotlib.pyplot as plt

colnames = ["SO2Concentration", "SurfaceRecessionRate"]
data = pd.read_csv("tombstone.dat", names=colnames, delim_whitespace=True)
x = data.SO2Concentration
y = data.SurfaceRecessionRate

xSum = 0
ySum = 0
for i in range(len(x)):
    xSum += x[i]
    ySum += y[i]

xMean = xSum/len(x)
yMean = ySum/len(y)

slopeNumerator = 0
slopeDenominator = 0
for i in range(len(x)):
    slopeNumerator += (x[i]-xMean)*(y[i]-yMean)
    slopeDenominator += (x[i] - xMean) * (x[i] - xMean)

#Results
slope = slopeNumerator/slopeDenominator
intercept = yMean-slope*xMean
yHat = slope * x + intercept
print("Intercept: ", intercept)
print("Slope/Coefficient : ", slope)
print("Prediction: ", yHat)

#Graph
plt.scatter(x, y)
plt.plot(x, yHat)
plt.title("Simple Linear Regression")
plt.xlabel("SO2 Concentration")
plt.ylabel("SurfaceRecessionRate")
plt.show()