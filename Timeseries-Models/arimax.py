import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from warpdrive import WarpDrive


wd = WarpDrive()

df = wd.get_args("df")
# your code here

def arimax_func(df,target,order):
  X = df.drop(columns=[target,'Date'])
  y = np.asarray(df[target])
  model =  sm.tsa.ARIMA(endog=y, order=order, exog=X)
  res_arimax = model.fit()
  return res_arimax

order = (5, 2, 0)  
arimax_result = arimax_func(df, 'Unemployment_rate', order)

exogenous_vars = ['RealGDPgrowth','NominalGDPgrowth']

wd.create_model(arimax_result,"statsmodels", "ARIMAX", "df", "Unemployment_rate", exogenous_vars)
