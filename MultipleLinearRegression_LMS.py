import pandas as pd
import numpy as np
import math

def GradientDescent(x, y):
    slopeCurr = np.random.uniform(0, 10)
    n = len(x)
    learningRate = 0.01
    i = 0
    prevLoss = 0

    while True:
        i += 1

        yHat = slopeCurr * x
        yHat = np.sum(yHat, axis=1)
        yHat = yHat.reshape(20,1)

        loss = (1 / n) * sum([val ** 2 for val in (y - yHat)])
        slopeDerivative = -(2/n)*sum(x*(y-yHat))
        slopeCurr = slopeCurr - learningRate * slopeDerivative
        print("Slopes: {}, Intercept: {}, Loss: {}, Iteration: {}".format(slopeCurr[1:], slopeCurr[0], loss, i))
        if math.isclose(loss, prevLoss, rel_tol=1e-20):
            break
        prevLoss = loss

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
print(y.shape)
GradientDescent(X, y)