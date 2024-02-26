import numpy as np
import pandas as pd

from numpy.linalg import det as np_det, inv as np_inv

from sklearn.cluster import KMeans as SklKMeans, SpectralClustering as SklSpectralClustering
from sklearn.ensemble import RandomForestClassifier as SklRandomForestClassifier
from sklearn.linear_model import LinearRegression as SklLinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.mixture import GaussianMixture as SklGaussianMixture
from sklearn.preprocessing import MinMaxScaler as SklMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklStandardScaler
from sklearn.preprocessing import PolynomialFeatures as SklPolynomialFeatures


def regression_error(labels, predicted):
  if not (isinstance(labels, pd.core.frame.DataFrame) or isinstance(labels, pd.core.series.Series)):
    raise Exception("truth labels has wrong type. Please use pandas DataFrame or Series")
  if not (isinstance(predicted, pd.core.frame.DataFrame) or isinstance(predicted, pd.core.series.Series)):
    raise Exception("predicted labels has wrong type. Please use pandas DataFrame or Series")

  return mean_squared_error(labels.values, predicted.values, squared=False)

def classification_error(labels, predicted):
  if not (isinstance(labels, pd.core.frame.DataFrame) or isinstance(labels, pd.core.series.Series)):
    raise Exception("truth labels has wrong type. Please use pandas DataFrame or Series")
  if not (isinstance(predicted, pd.core.frame.DataFrame) or isinstance(predicted, pd.core.series.Series)):
    raise Exception("predicted labels has wrong type. Please use pandas DataFrame or Series")

  return 1.0 - accuracy_score(labels.values, predicted.values)

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


class Predictor():
  def __init__(self, type, **kwargs):
    if type == "linear":
      self.model = SklLinearRegression(**kwargs)
    elif type == "class":
      if "max_depth" not in kwargs:
        kwargs["max_depth"]=16
      self.model = SklRandomForestClassifier(**kwargs)

  def fit(self, X, y, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    if not (isinstance(y, pd.core.frame.DataFrame) or isinstance(y, pd.core.series.Series)):
      raise Exception("Label input has wrong type. Please use pandas DataFrame or Series")

    self.y_name = y.name if len(y.shape) == 1 else y.columns[0]
    self.model.fit(X.values, y.values, *args, **kwargs)

  def predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Feature input has wrong type. Please use pandas DataFrame")
    y_t = self.model.predict(X.values, *args, **kwargs)
    return pd.DataFrame(y_t, columns=[self.y_name])


class Scaler():
  def __init__(self, type, **kwargs):
    if type == "minmax":
      self.scaler = SklMinMaxScaler(**kwargs)
    elif type == "std":
      self.scaler = SklStandardScaler(**kwargs)

  def fit_transform(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")

    self.columns = X.columns
    self.shape = X.shape
    X_t = self.scaler.fit_transform(X.values, *args, **kwargs)
    return pd.DataFrame(X_t, columns=X.columns)

  def transform(self, X, *args, **kwargs):
    if type(X) == np.ndarray:
      return self.scaler.transform(X, *args, **kwargs)

    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")
    if list(self.columns) != list(X.columns) or self.shape[1] != X.shape[1]:
      raise Exception("Input has wrong shape.")

    X_t = self.scaler.transform(X.values, *args, **kwargs)
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
      X_t = self.scaler.inverse_transform(dummy_df.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=self.columns)[[col]]

    else:
      X_t = self.scaler.inverse_transform(X.values, *args, **kwargs)
      return pd.DataFrame(X_t, columns=X.columns)


class Clusterer():
  def __init__(self, type, **kwargs):
    self.num_clusters = 0
    kwargs["n_init"] = 10
    if type == "kmeans":
      self.model = SklKMeans(**kwargs)
    elif type == "gaussian":
      if "n_clusters" in kwargs:
        kwargs["n_components"] = kwargs["n_clusters"]
        del kwargs["n_clusters"]
      self.model = SklGaussianMixture(**kwargs)
    elif type == "spectral":
      if "affinity" not in kwargs:
        kwargs["affinity"] = 'nearest_neighbors'
      if "n_clusters" in kwargs:
        kwargs["n_clusters"] += 1
      self.model = SklSpectralClustering(**kwargs)

  def fit_predict(self, X, *args, **kwargs):
    if not isinstance(X, pd.core.frame.DataFrame):
      raise Exception("Input has wrong type. Please use pandas DataFrame")

    y = self.model.fit_predict(X.values, *args, **kwargs)
    self.X = X.values
    self.y = y
    self.num_clusters = len(np.unique(y))
    self.num_features = self.X.shape[1]
    return pd.DataFrame(y, columns=["clusters"])

  def distance_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")

    centers = np.array([self.X[self.y == c].mean(axis=0) for c in range(self.num_clusters)])
    point_centers = [centers[i] for i in self.y]
    point_diffs = np.array([p - c for p,c in zip(self.X, point_centers)])

    cluster_L2 = [np.sqrt(np.square(point_diffs[self.y == c]).sum(axis=1)).mean() for c in range(self.num_clusters)]

    return sum(cluster_L2) / len(cluster_L2)

  def likelihood_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
    means = np.array([self.X[self.y == c].mean(axis=0) for c in range(self.num_clusters)])
    covs = np.array([np.cov(self.X[self.y==c].transpose()) for c in range(self.num_clusters)])

    point_means = [means[i] for i in self.y]
    point_covs = [covs[i] for i in self.y]

    two_pi_term = np.power(2 * np.pi, self.num_features)

    point_density_den = [np.sqrt(two_pi_term * np_det(cov)) for cov in point_covs]
    point_density_num = [np.exp(-0.5 * (p - m) @ np_inv(cov) @ (p - m)) for p,m,cov in zip(self.X, point_means, point_covs)]
    point_density = np.array(point_density_num) / np.array(point_density_den)

    cluster_log_like = [np.log(point_density[self.y == c]).mean() for c in range(self.num_clusters)]

    return sum(cluster_log_like) / len(cluster_log_like)

  def balance_error(self):
    if self.num_clusters < 1:
      raise Exception("Error: need to run fit_predict() first")
    counts = np.unique(self.y, return_counts=True)[1]
    return np.abs(counts / len(self.y) - (1 / self.num_clusters)).max()

class LinearRegression(Predictor):
  def __init__(self, **kwargs):
    super().__init__("linear", **kwargs)

class RandomForestClassifier(Predictor):
  def __init__(self, **kwargs):
    super().__init__("class", **kwargs)

class MinMaxScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("minmax", **kwargs)

class StandardScaler(Scaler):
  def __init__(self, **kwargs):
    super().__init__("std", **kwargs)

class KMeansClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("kmeans", **kwargs)

class GaussianClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("gaussian", **kwargs)

class SpectralClustering(Clusterer):
  def __init__(self, **kwargs):
    super().__init__("spectral", **kwargs)
