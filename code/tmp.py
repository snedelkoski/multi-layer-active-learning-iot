import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

print (np.load('../outputs/synt/model_size_90.npy'))
df = pd.read_csv('../datasets/parkinsons_updrs.csv')
y = df.total_UPDRS
df.drop(['subject#','age','sex','total_UPDRS','motor_UPDRS'], axis=1)

X = minmax_scale(df.values)
pf = PolynomialFeatures(degree=2)
print (X.shape)
X = pf.fit_transform(X)

print(X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

from sklearn.linear_model import BayesianRidge