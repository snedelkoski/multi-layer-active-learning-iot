import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets.dataset import *

def random_sampling(nr_exp=0, m=50, accuracy=None):
    create_classification_data(n_samples=550, n_features=90, n_informative=18, class_sep=1.8)
    error_history_rs = []
    x_unlabeled = np.load("../datasets/xunlabeled.npy", mmap_mode='r')
    y_unlabeled = np.load("../datasets/yunlabeled.npy", mmap_mode='r')

    x_test = np.load("../datasets/xtest.npy", mmap_mode='r')
    y_test = np.load("../datasets/ytest.npy", mmap_mode='r')

    x_labeled = np.load("../datasets/xlabeled.npy", mmap_mode='r')
    y_labeled = np.load("../datasets/ylabeled.npy", mmap_mode='r')

    clf = RandomForestClassifier(n_estimators=50, max_depth=5)
    clf.fit(x_unlabeled, y_unlabeled)
    x_predict = clf.predict(x_test)
    error_history_rs.append(accuracy_score(x_predict, y_test))
    print(accuracy_score(x_predict, y_test))

    for i in range(0, len(x_unlabeled)):

        #if accuracy is not None and len(error_history_rs) > 0:
            #print("ACC, RANDOM sampling currently", len(error_history_rs),
            #      "number of samples with accuracy:", error_history_rs[-1])

        #if accuracy is None and len(error_history_rs) > 0:
            #print("RANDOM sampling currently", len(error_history_rs),
            #      "number of samples with accuracy:", error_history_rs[-1])

        random_index = np.random.randint(i, i + len(x_unlabeled) // m)
        sample = x_unlabeled[i:i + 1]
        y_sample = y_unlabeled[i:i + 1]
        x_labeled = np.append(x_labeled, sample, axis=0)
        y_labeled = np.append(y_labeled, y_sample, axis=0)
        clf = RandomForestClassifier(n_estimators=50, max_depth=5)
        clf.fit(x_labeled, y_labeled)
        x_predict = clf.predict(x_test)
        error_history_rs.append(accuracy_score(x_predict, y_test))
        print(i, accuracy_score(x_predict, y_test))
        if accuracy_score(x_predict, y_test) > 0.89:
            #print (i, accuracy_score(x_predict, y_test))
            break
    np.save("../outputs/paper_results/fallfull.csv", np.array(error_history_rs))

random_sampling()