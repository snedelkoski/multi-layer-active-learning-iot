from datasets.dataset import create_classification_data
import time
from multiprocessing import Process
import numpy as np
from pathlib import Path
from code.random_sampling import random_sampling
from itertools import product
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    n_samples = 300000
    device_sampling_rate = 0.01
    nr_exp = 0
    M = 50

    np.save('../outputs/sampling_rate.npy', np.float32(device_sampling_rate))
    np.save('../outputs/nr_exp.npy', np.int32(nr_exp))
    np.save('../outputs/kill.npy', np.int32(0))

    latency = 0
    bandwidth = [1000, 25000, 125000]
    nr_devices = [1, 4, 8, 12]
    budget = [0.4, 0.8, 2]
    n_features = [150]
    n_informative = [x // 5 for x in n_features]

    for nf, ni in zip(n_features, n_informative):
        for i in range(2):
            print("started creating data..")
            create_classification_data(n_samples=n_samples, n_features=nf, n_informative=ni, class_sep=1.8)
            print("data created")
            for nd, b, band in product(nr_devices, budget, bandwidth):
                import code.trainserver as trainserver
                import code.edge as edge
                import code.device as device

                print("###############################################################")
                print("Running experiment number", nr_exp, "with following parameters:")
                print("###############################################################")
                print("Number of features:", nf)
                print("Number of informative features:", ni)
                print("Number of devices:", nd)
                print("Budget in seconds:", b)
                print("Bandwidth:", band)
                print("M=", M)


                p1 = Process(target=trainserver.start_server, args=(nr_exp, i, b, M))
                p1.start()
                print("server started")
                time.sleep(5)

                p2 = Process(target=edge.start_edge, args=(nr_exp, band, latency))
                p2.start()
                print("edge started")
                time.sleep(5)

                p3 = Process(target=device.start_devices, args=(nd,))
                p3.start()
                print("devices started")

                p4 = Process(target=random_sampling, args=(nr_exp, M))
                p4.start()
                print("random sampling started")

                while True:
                    my_file_us = Path("../outputs/" + str(nr_exp) + "_try_" + str(i) + "_historyUS_.npy")
                    my_file_rs = Path("../outputs/" + str(nr_exp) + "_rs.npy")
                    if my_file_us.is_file() and my_file_rs.is_file():
                        break
                    time.sleep(5)

                np.save('../outputs/kill.npy', np.int32(1))
                time.sleep(4)
                p1.terminate()
                time.sleep(1)
                p2.terminate()
                time.sleep(1)
                p3.terminate()
                time.sleep(1)

                nr_exp += 1
                np.save('../outputs/kill.npy', np.int32(0))
                np.save('../outputs/nr_exp.npy', np.int32(nr_exp))
