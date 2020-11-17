import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import os
import scipy.io as sio
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_classification_data(n_samples=300000, n_features=150, n_informative=30, class_sep=1.8):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, class_sep=class_sep, weights=[0.8, 0.2])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)
    xlabeled, xunlabeled, ylabeled, yunlabeled = train_test_split(x_train, y_train, train_size=10, random_state=15, stratify=y_train)
    print ("A")
    np.save("../datasets/xlabeled.npy", xlabeled)
    np.save("../datasets/xunlabeled.npy", xunlabeled)
    np.save("../datasets/ylabeled.npy", ylabeled)
    np.save("../datasets/yunlabeled.npy", yunlabeled)
    np.save("../datasets/xtest.npy", x_test)
    np.save("../datasets/ytest.npy", y_test)


def zc(arr):
    return np.count_nonzero(np.where(np.diff(np.sign(arr)))[0])

def extract_features(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    count_zero_crossings = zc(arr)
    ma = np.mean(np.diff(arr))
    hist = np.histogram(minmax_scale(arr), bins=np.arange(0, 1.1, 0.1))
    features = np.concatenate(([mean], [std], [count_zero_crossings], [ma], hist[0]))
    return features

def read_fall(mypath='../datasets/UMAFall_Dataset/'):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    features = []
    y = []
    for file in onlyfiles:
        data = pd.read_csv(mypath+file, header=32, sep=';')
        try:
            x_by_id = data.groupby(by=' Sensor ID')[' X-Axis'].apply(np.array).reset_index()
            y_by_id = data.groupby(by=' Sensor ID')[' Y-Axis'].apply(np.array).reset_index()
            z_by_id = data.groupby(by=' Sensor ID')[' Z-Axis'].apply(np.array).reset_index()
        except KeyError:
            continue
        feats = []
        if (len(x_by_id[' X-Axis'].values)>4):
            for i in range(len(x_by_id[' X-Axis'].values)):
                x_feats = extract_features(x_by_id[' X-Axis'].values[i])
                y_feats = extract_features(y_by_id[' Y-Axis'].values[i])
                z_feats = extract_features(z_by_id[' Z-Axis'].values[i])
                feats.append(x_feats)
                feats.append(y_feats)
                feats.append(z_feats)
            if '_Fall_' in file:
                y.append(1)
            else:
                y.append(0)
            #y.append()
            features.append(np.array(feats).flatten())
    features = np.array(features)
    y = np.array(y)
    features = np.repeat(features, 200, axis=0)
    y = np.repeat(y, 200, axis=0)

    x_train, x_test, y_train, y_test = train_test_split(features, y, train_size=0.8, stratify=y)
    xlabeled, xunlabeled, ylabeled, yunlabeled = train_test_split(x_train, y_train, train_size=2, random_state=15, stratify=y_train)
    np.save("../datasets/xlabeled.npy", xlabeled)
    np.save("../datasets/xunlabeled.npy", xunlabeled)
    np.save("../datasets/ylabeled.npy", ylabeled)
    np.save("../datasets/yunlabeled.npy", yunlabeled)
    np.save("../datasets/xtest.npy", x_test)
    np.save("../datasets/ytest.npy", y_test)


def load_ecg(mypath='../datasets/ecg/'):

    X = np.load('x.npy')
    y = np.load('y.npy')

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
    #X = np.repeat(x_train, 15, axis=0)
    #y = np.repeat(y_train, 15, axis=0)
    rnd = np.random.normal(0,0.1, (X.shape[0], X.shape[1]))
    X = np.add(X, rnd)
    values = [i for i in range(len(X))]
    permutations = np.random.permutation(values)
    X = X[permutations, :]
    y = y[permutations]
    xlabeled, xunlabeled, ylabeled, yunlabeled = train_test_split(X, y, train_size=2, random_state=15,
                                                                  stratify=y)

    print (xunlabeled.shape)
    np.save("../datasets/xlabeled.npy", xlabeled)
    np.save("../datasets/xunlabeled.npy", xunlabeled)
    np.save("../datasets/ylabeled.npy", ylabeled)
    np.save("../datasets/yunlabeled.npy", yunlabeled)
    np.save("../datasets/xtest.npy", x_test)
    np.save("../datasets/ytest.npy", y_test)

def load_reg(mypath='../datasets/'):
    df = pd.read_csv('../datasets/parkinsons_updrs.csv')
    y = df.total_UPDRS
    df.drop(['subject#', 'age', 'sex', 'total_UPDRS', 'motor_UPDRS'], axis=1)

    X = minmax_scale(df.values)
    pf = PolynomialFeatures(degree=1)
    X = pf.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    for i in range(5):
        x_train = np.append(x_train, x_train, axis=0)
        y_train = np.append(y_train, y_train, axis=0)
    rnd = np.random.normal(0, 0.005, (x_train.shape[0], x_train.shape[1]))
    x_train = np.add(x_train, rnd)
    values = [i for i in range(len(x_train))]
    permutations = np.random.permutation(values)
    x_train = x_train[permutations, :]
    y_train = y_train[permutations]
    xlabeled, xunlabeled, ylabeled, yunlabeled = train_test_split(x_train, y_train, train_size=100, random_state=15)
    np.save("../datasets/xlabeled.npy", xlabeled)
    np.save("../datasets/xunlabeled.npy", xunlabeled)
    np.save("../datasets/ylabeled.npy", ylabeled)
    np.save("../datasets/yunlabeled.npy", yunlabeled)
    np.save("../datasets/xtest.npy", x_test)
    np.save("../datasets/ytest.npy", y_test)

read_fall()