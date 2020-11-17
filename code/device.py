import multiprocessing
import code.utilities as utilities
import numpy as np
import time
import pickle
from pympler import asizeof


def device(index):
    x_unlabeled = np.load("../datasets/xunlabeled.npy")
    sampling_rate = float(np.load('../outputs/sampling_rate.npy'))
    nr_exp = int(np.load('../outputs/nr_exp.npy'))
    host = 'localhost'
    port = 11111
    s = pickle.dumps(x_unlabeled[0:0 + 1])
    size_of_sample = asizeof.asizeof(s)
    np.save('../outputs/' + str(nr_exp) + '_size_of_sample' + str(x_unlabeled.shape[0]) + '_' + str(
        x_unlabeled.shape[1]) + '.npy', np.int(size_of_sample))
    my_socket = utilities.create_socket('normal', host, port)
    buff_size = 4096
    print(multiprocessing.current_process(), "starts sending data")
    for i in range(index[0], index[1]):
        time.sleep(sampling_rate)
        x_sending_sample = x_unlabeled[i:i + 1]
        s = pickle.dumps((i, x_sending_sample))
        try:
            my_socket.send(s)
        except ConnectionResetError:
            break
        try:
            my_socket.recv(buff_size)
        except ConnectionResetError:
            break
        time.sleep(sampling_rate)

    my_socket.close()


def start_devices(nr_devices=4):
    x_unlabeled = np.load("../datasets/xunlabeled.npy")
    pool = multiprocessing.Pool(processes=nr_devices)
    split = np.linspace(0, len(x_unlabeled), nr_devices + 1, dtype=np.int32)
    args = []
    for i in range(nr_devices):
        args.append((split[i], split[i + 1]))
    pool.map(device, args)
    while True:
        time.sleep(1)
        kill = int(np.load('../outputs/kill.npy'))
        if kill:
            pool.terminate()
            np.save('../outputs/kill.npy', np.int32(0))
