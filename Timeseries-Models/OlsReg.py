import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from warpdrive import WarpDrive


wd = WarpDrive()

df = wd.get_args("df")
# your code here

X = df[["Mortgage_rate","RealGDPgrowth"]]
print(X.shape)
y = np.asarray(df['Unemployment_rate'])

model = sm.OLS(y,X) 
ols_model = model.fit()

exogenous_vars = ["Mortgage_rate"]

wd.create_model(ols_model, "statsmodels","OLS REGRESSION", "df", "Unemployment_rate", exogenous_vars)
