import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
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

def read_data(mypath):
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
    x_train, x_test, y_train, y_test = train_test_split(features, y, train_size=0.8, stratify=y)
    xlabeled, xunlabeled, ylabeled, yunlabeled = train_test_split(x_train, y_train, train_size=10, random_state=15, stratify=y_train)
    np.save("../datasets/xlabeled.npy", xlabeled)
    np.save("../datasets/xunlabeled.npy", xunlabeled)
    np.save("../datasets/ylabeled.npy", ylabeled)
    np.save("../datasets/yunlabeled.npy", yunlabeled)
    np.save("../datasets/xtest.npy", x_test)
    np.save("../datasets/ytest.npy", y_test)

    print (xunlabeled.shape)
    print (ylabeled)

    return features, y


read_data('../datasets/UMAFall_Dataset/')


