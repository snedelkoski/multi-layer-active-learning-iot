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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_ecg(mypath='../datasets/ecg/'):
    from keras.models import Model
    from keras.models import load_model
    from keras.models import Model

    number_of_classes = 2
    mypath = '../datasets/ecg/'  # Training directory
    onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f[0] == 'A')]
    bats = [f for f in onlyfiles if f[7] == 'm']
    check = 100
    mats = [f for f in bats if (np.shape(sio.loadmat(mypath + f)['val'])[1] >= check)]
    size = len(mats)
    print('Total training size is ', size)
    big = 10100
    X = np.zeros((size, big))
    ######Old stuff
    # for i in range(size):
    # X[i, :] = sio.loadmat(mypath + mats[i])['val'][0, :check]
    ######

    for i in range(size):
        dummy = sio.loadmat(mypath + mats[i])['val'][0, :]
        if (big - len(dummy)) <= 0:
            X[i, :] = dummy[0:big]
        else:
            b = dummy[0:(big - len(dummy))]
            goal = np.hstack((dummy, b))
            while len(goal) != big:
                b = dummy[0:(big - len(goal))]
                goal = np.hstack((goal, b))
            X[i, :] = goal

    target_train = np.zeros((size, 1))
    Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
    for i in range(size):
        if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
            target_train[i] = 0
        elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
            target_train[i] = 1
        else:
            target_train[i] = 2

    X = X[target_train.flatten() != 2]
    target_train = target_train[target_train.flatten() != 2]
    size = len(X)
    Label_set = np.zeros((size, number_of_classes))
    for i in range(size):
        dummy = np.zeros((number_of_classes))
        dummy[int(target_train[i])] = 1
        Label_set[i, :] = dummy

    X = (X - X.mean()) / (X.std())  # Some normalization here
    X = np.expand_dims(X, axis=2)  # For Keras's data input size

    model = load_model('../code/Conv_models/Best_model.h5')

    layer_name = 'l1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    X = intermediate_layer_model.predict(X)
    print (X.shape)
    #X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    y = np.argmax(Label_set, axis=1).flatten()
    np.save('x.npy',X)
    np.save('y.npy',y)


load_ecg()