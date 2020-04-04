import pandas as pd
import numpy as np

colnames = ["TreatmentNumber", "ObservationNumber", "CumulativeAverageCost",
            "CumulativeProduction", "CumulativeTrainingTime", "lnY", "lnX1", "lnX2"]
data = pd.read_csv("manuf_learn.dat", names=colnames, delim_whitespace=True)

X1 = np.array(data.lnX1)
X2 = np.array(data.lnX2)
X0 = np.ones(X1.shape)
#Merge X0 X1 X2
X = np.column_stack((X0, X1, X2))
y = np.array(data.lnY)
y = y.reshape(len(y), 1)
slope = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)
yHat = X.dot(slope)

#Results
print("Slopes/Coefficients: ", slope[1:])
print("Intercept: ", slope[0])
print("Prediction: ", yHat)