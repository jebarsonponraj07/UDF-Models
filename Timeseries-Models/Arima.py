import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from warpdrive import WarpDrive
 
# Initialize WarpDrive
wd = WarpDrive()
 
df = wd.get_args("df")
 
def arima_func(df):
  model = sm.tsa.ARIMA(df.Unemployment_rate, order=(2,3,4)) 
  res_arima = model.fit()
  return res_arima
 
arima_result = arima_func(df)
col=df.drop(columns='Unemployment_rate').columns.tolist()
wd.create_model(arima_result,"statsmodels", "ARIMA", "df",col,"Unemployment_rate")
