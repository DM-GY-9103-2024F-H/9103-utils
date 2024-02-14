import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.preprocessing import MinMaxScaler as SklMinMaxScaler


class MinMaxScaler(SklMinMaxScaler):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def fit_transform(self, X, *args, **kwargs):
    if isinstance(X, pd.core.frame.DataFrame):
      _X = super().fit_transform(X.values, *args, **kwargs)
      return pd.DataFrame(_X, columns=X.columns)
    else:
      return super().fit_transform(X, *args, **kwargs)

  def inverse_transform(self, X, *args, **kwargs):
    if isinstance(X, pd.core.frame.DataFrame):
      _X = super().inverse_transform(X.values, *args, **kwargs)
      return pd.DataFrame(_X, columns=X.columns)
    else:
      return super().fit_transform(X, *args, **kwargs)
