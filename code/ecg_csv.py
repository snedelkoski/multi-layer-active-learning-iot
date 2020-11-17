from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, \
    GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers

np.random.seed(7)

number_of_classes = 2  # Total number of classes


def change(x):  # From boolean arrays to decimal arrays
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


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

X = X[target_train.flatten()!=2]
target_train = target_train[target_train.flatten()!=2]
size = len(X)
Label_set = np.zeros((size, number_of_classes))
for i in range(size):
    dummy = np.zeros((number_of_classes))
    dummy[int(target_train[i])] = 1
    Label_set[i, :] = dummy

X = (X - X.mean()) / (X.std())  # Some normalization here
X = np.expand_dims(X, axis=2)  # For Keras's data input size
print (len(X), len(Label_set))
values = [i for i in range(size)]
permutations = np.random.permutation(values)
X = X[permutations, :]
Label_set = Label_set[permutations, :]

train = 0.9  # Size of training set in percentage
X_train = X[:int(train * size), :]
Y_train = Label_set[:int(train * size), :]
X_val = X[int(train * size):, :]
Y_val = Label_set[int(train * size):, :]

# def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i):

# def create_model():
model = Sequential()
model.add(Conv1D(32, 55, activation='relu', input_shape=(big, 1)))
model.add(MaxPooling1D(10))
model.add(Dropout(0.5))
model.add(Conv1D(32, 25, activation='relu', name='l'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))
model.add(Conv1D(32, 10, activation='relu'))
model.add(GlobalAveragePooling1D(name='l0'))
model.add(Dense(256, kernel_initializer='normal', activation='relu', name='l1'))
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer='normal', activation='relu', name='l2'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='normal', activation='relu', name='l3'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='Conv_models/Best_model.h5', monitor='val_acc', verbose=1, save_best_only=True)
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=50, epochs=10, verbose=1, shuffle=True,
                 callbacks=[checkpointer])

pd.DataFrame(hist.history).to_csv(path_or_buf='Conv_models/History.csv')
predictions = model.predict(X_val)
score = accuracy_score(change(Y_val), change(predictions))
print('Last epoch\'s validation score is ', score)
df = pd.DataFrame(change(predictions))
df.to_csv(path_or_buf='Conv_models/Preds_' + str(format(score, '.4f')) + '.csv', index=None, header=None)
pd.DataFrame(confusion_matrix(change(Y_val), change(predictions))).to_csv(
    path_or_buf='Conv_models/Result_Conf' + str(format(score, '.4f')) + '.csv', index=None, header=None)