import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier as SklRandomForestClassifier
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.preprocessing import MinMaxScaler as SklMinMaxScaler
from sklearn.preprocessing import PolynomialFeatures as SklPolynomialFeatures


class RandomForestClassifier(SklRandomForestClassifier):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit(self, X, y, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if not (isinstance(y, pd.core.frame.DataFrame) or isinstance(y, pd.core.series.Series)):
      raise Exception("Label input has wrong type. Please use pandas DataFrame or Series")

    self.y_name = y.name if len(y.shape) == 1 else y.columns[0]
    super().fit(X.values, y.values, *args, **kwargs)

  def predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    y_t = super().predict(X.values, *args, **kwargs)
    return pd.DataFrame(y_t, columns=[self.y_name])


class PolynomialFeatures(SklPolynomialFeatures):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit_transform(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")

    self.columns = X.columns
    self.shape = X.shape
    X_t = super().fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())

  def transform(self, X, *args, **kwargs):
    if type(X) == np.ndarray:
      return super().transform(X, *args, **kwargs)

    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = super().transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=self.get_feature_names_out())


class LinearRegression(SklLinearRegression):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def fit(self, X, y, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if not (isinstance(y, pd.core.frame.DataFrame) or isinstance(y, pd.core.series.Series)):
      raise Exception("Label input has wrong type. Please use pandas DataFrame or Series")

    self.y_name = y.name if len(y.shape) == 1 else y.columns[0]
    super().fit(X.values, y.values, *args, **kwargs)

  def predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    y_t = super().predict(X.values, *args, **kwargs)
    return pd.DataFrame(y_t, columns=[self.y_name])


class MinMaxScaler(SklMinMaxScaler):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  def fit_transform(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")

    self.columns = X.columns
    self.shape = X.shape
    X_t = super().fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def transform(self, X, *args, **kwargs):
    if type(X) == np.ndarray:
      return super().transform(X, *args, **kwargs)

    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")
    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = super().transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def inverse_transform(self, X, *args, **kwargs):
    if not (isinstance(X, pd.core.frame.DataFrame) or isinstance(X, pd.core.series.Series)):
      raise Exception("Input has wrong type. Please use pandas DataFrame or Series")

    col = ""
    col_vals = []

    if len(X.shape) == 1:
      col = X.name
      col_vals = X.values
    elif len(X.shape) == 2 and X.shape[1] == 1:
      col = X.columns[0]
      col_vals = X[col].values

    if col != "":
      X = pd.DataFrame(X.values, columns=[col])
      dummy_df = pd.DataFrame(np.zeros((len(col_vals), self.shape[1])), columns=self.columns)
      dummy_df[col] = col_vals
      X_t = super().inverse_transform(dummy_df.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=self.columns)[[col]]

    else:
      X_t = super().inverse_transform(X.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=X.columns)
