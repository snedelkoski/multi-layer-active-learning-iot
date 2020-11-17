import _thread
import threading
from sklearn.ensemble import RandomForestClassifier
import code.utilities as utilities
import numpy as np
import pickle


def model():
    print("thread for the model is running")
    global clf
    port = 33333
    host = 'localhost'
    backlog = 20
    buf_size = 4096
    listening_socket = utilities.create_socket('listen', host, port, backlog)
    c, address = listening_socket.accept()
    print("Address connected for model transfer", address)
    while True:
        tmp = utilities.receive_sample(c, buf_size)
        if not isinstance(tmp, int):
            clf = tmp


def active_choose_sample(buffer_data):
    global clf
    x = [bd[1] for bd in buffer_data]
    idx = [bd[0] for bd in buffer_data]
    x = np.array(x)
    x = x.reshape((x.shape[0], x.shape[2]))
    preds = clf.predict_proba(x)
    most_uncertain_idx = np.argmin(np.abs(0.5 - preds[:, 0]))
    return idx[most_uncertain_idx], x[most_uncertain_idx:most_uncertain_idx + 1]


def active_send(my_socket, buf_size, myfile):
    global buffer
    len_buffer = len(buffer)
    len_buffer = len_buffer + 1 - 1
    if len_buffer > 0 and clf is not None:
        buffer_data = buffer[:len_buffer]
        del buffer[:len_buffer]
        sample_to_send = active_choose_sample(buffer_data)
        try:
            my_socket.send(pickle.dumps(sample_to_send))
        except ConnectionResetError:
            return

        my_socket.recv(buf_size)
        with open(myfile, "a") as f:
            f.write(str(len_buffer) + '\n')


def edge_send(nr_exp, bandwidth=2000, latency=0):
    global buffer
    port = 22222
    host = 'localhost'
    buf_size = 1024

    my_socket = utilities.create_socket('control', host, port, [latency, bandwidth])
    myfile = "../outputs/" + str(nr_exp) + "_l_" + str(latency) + "_b_" + str(bandwidth) + ".txt"
    with open(myfile, "w") as f:
        f.write("len_buffer_edge\n")

    while True:
        active_send(my_socket, buf_size, myfile)


def on_new_client(accepted_socket, buf_size):
    global buffer

    while True:
        data = utilities.receive_sample(accepted_socket, buf_size)
        if not isinstance(data, int):
            buffer.append(data)
            try:
                accepted_socket.send("received".encode())
            except ConnectionResetError:
                break


def edge_receive():
    global buffer
    port = 11111
    host = 'localhost'
    backlog = 20
    buf_size = 4096

    listening_socket = utilities.create_socket('listen', host, port, backlog)

    while True:
        c, address = listening_socket.accept()
        print("Address connected", address)
        _thread.start_new_thread(on_new_client, (c, buf_size))


def start_edge(nr_exp=0, bandwidth=2000, latency=0):
    global clf
    clf = None
    global buffer
    buffer = []
    threading.Thread(target=edge_receive).start()
    threading.Thread(target=edge_send, args=(nr_exp, bandwidth, latency)).start()
    threading.Thread(target=model).start()


global buffer
global clf
clf = None
buffer = []
