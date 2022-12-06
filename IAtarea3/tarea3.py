
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


data = pd.read_csv("dataset.csv")
data.describe()
data.head()