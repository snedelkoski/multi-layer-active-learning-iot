import code.utilities as utilities
import warnings
import numpy as np
from sklearn.linear_model import BayesianRidge
import time
import threading
import pickle
from sklearn.metrics import mean_squared_error
from pympler import asizeof

warnings.filterwarnings('ignore')


def update_model(nr_exp, budget):
    global x_unlabeled, y_unlabeled, x_labeled, y_labeled, x_test, y_test, clf, server_buffer, errorHistoryUS

    if clf is None:
        clf = BayesianRidge()
        clf.fit(x_labeled, y_labeled)
        np.save('../outputs/model_size_' + str(x_unlabeled.shape[1]) + '.npy',
                np.int32(asizeof.asizeof(pickle.dumps(clf))))
        print (np.int32(asizeof.asizeof(pickle.dumps(clf))))
    host = 'localhost'
    port = 33333
    my_socket = -1
    while my_socket == -1:
        my_socket = utilities.create_socket('normal', host, port)

    print("Connection to edge for model sharing established")
    my_socket.send(pickle.dumps(clf))
    myfile = "../outputs/" + str(nr_exp) + "_budget_" + str(budget) + ".txt"
    with open(myfile, "w") as f:
        f.write("len_buffer_server\n")
    while True:
        time.sleep(budget)
        len_buffer = len(server_buffer)
        if len_buffer > 0:
            buffer_data = server_buffer[:len_buffer]
            del server_buffer[:len_buffer]
            x = [bd[1] for bd in buffer_data]
            idx = [bd[0] for bd in buffer_data]
            x = np.array(x)
            x = x.reshape((x.shape[0], x.shape[2]))
            _, std = clf.predict(x, return_std=True)
            most_uncertain_idx = np.argmax(std)
            x_labeled = np.append(x_labeled, x[most_uncertain_idx:most_uncertain_idx + 1], axis=0)
            y_labeled = np.append(y_labeled, y_unlabeled[idx[most_uncertain_idx]])
            clf = BayesianRidge()
            clf.fit(x_labeled, y_labeled)
            p = clf.predict(x_test)
            errorHistoryUS.append(np.sqrt(mean_squared_error(y_test.flatten(), p.flatten())))
            with open(myfile, "a") as f:
                f.write(str(len_buffer) + '\n')
        try:
            my_socket.send(pickle.dumps(clf))
        except ConnectionResetError:
            break


def server():
    global server_buffer
    port = 22222
    host = 'localhost'
    backlog = 5
    buf_size = 4096

    listening_socket = utilities.create_socket('listen', host, port, backlog)

    accepted_socket, address = listening_socket.accept()
    print(address, 'Edge is connected!')

    while True:

        data = utilities.receive_sample(accepted_socket, buf_size)
        if not isinstance(data, int):
            server_buffer.append(data)
        try:
            accepted_socket.send('Hello edge, I received the sample.'.encode())
        except ConnectionResetError:
            break


def start_server(nr_exp=0, i=0, budget=2, m=50, accuracy=None):
    global x_unlabeled, y_unlabeled, x_labeled, y_labeled, x_test, y_test, clf, server_buffer
    clf = None
    global errorHistoryUS
    errorHistoryUS = []
    x_unlabeled = np.load("../datasets/xunlabeled.npy")
    y_unlabeled = np.load("../datasets/yunlabeled.npy")

    x_test = np.load("../datasets/xtest.npy")
    y_test = np.load("../datasets/ytest.npy")

    x_labeled = np.load("../datasets/xlabeled.npy")
    y_labeled = np.load("../datasets/ylabeled.npy")

    server_buffer = []
    threading.Thread(target=server).start()
    threading.Thread(target=update_model, args=(nr_exp, budget)).start()
    while True:
        time.sleep(2)
        len_error = len(errorHistoryUS)
        if accuracy is not None and len_error > 0:
            print("UNCERTAINTY: The expert labeled", len_error,
                  "samples, with current accuracy of:", errorHistoryUS[len_error - 1])
            if errorHistoryUS[len_error - 1] > accuracy:
                break
        elif len_error > 0:
            print("UNCERTAINTY: The expert labeled", len_error,
                  "samples, with current accuracy of:", errorHistoryUS[len_error - 1])
            if len_error >= m:
                break

    np.save("../outputs/" + str(nr_exp) + "_try_" + str(i) + "_historyUS_.npy", np.array(errorHistoryUS[:m]))


global x_unlabeled, y_unlabeled, x_labeled, y_labeled, x_test, y_test, clf, server_buffer
clf = None
global errorHistoryUS
errorHistoryUS = []
