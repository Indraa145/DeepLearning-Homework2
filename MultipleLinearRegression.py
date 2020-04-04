from sklearn.linear_model import LinearRegression
import pandas as pd

colnames = ["StationCallLetters", "SalePrice", "MarketRetailSales", "MarketBuyingIncome",
            "TVHomesStation", "NetworkHourlyRate", "NationalSpotRate", "AgeOfStation",
            "NumberOfTiesWithMajorNetwork", "PercentOfMarketPopulationInUrbanAreas"]
data = pd.read_csv("tvsales.dat", names=colnames, delim_whitespace=True)

xColnames = colnames[2:]
model = LinearRegression()
model.fit(data[xColnames], data.SalePrice)

#Results
slopes = model.coef_
intercept = model.intercept_
# yHat = slope1 * x1 + slope2 * x2 + slope3 * x3 + ... + slopen * xn + intercept
yHat = model.predict(data[xColnames])
print("Slopes/Coefficients: ", slopes)
print("Intercept: ", intercept)
print("Predicted: ", yHat)
