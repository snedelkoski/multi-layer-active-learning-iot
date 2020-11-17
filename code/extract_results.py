import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from ast import literal_eval

def extract_results(my_path, file_name):
    files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    file_idx = []
    df = pd.DataFrame()
    for f in files:
        try:
            idx = int(f.split("_")[0])
            if idx not in file_idx:
                file_idx.append(idx)
        except ValueError:
            pass

    file_idx = np.sort(file_idx)

    for idx in file_idx:
        idx_files = []
        for f in files:
            try:
                f_idx = int(f.split("_")[0])
                if f_idx == idx:
                    idx_files.append(f)
            except ValueError:
                continue

        for f in idx_files:
            try:
                if "size" in f:
                    split_1 = f.split('.')
                    split_2 = split_1[0].split('_')
                    number_features = float(split_2[-1])
                    sample_size = float(np.load(my_path+f))

                if "rs" in f:
                    rs = np.load(my_path+f)

                if "US" in f:
                    us = np.load(my_path+f)

                if "budget" in f:
                    budget = float(os.path.splitext(f)[0].split('_')[-1])
                    len_buffer_server = int(np.mean(pd.read_csv(my_path+f).values))

                if "l_" in f:
                    split_1 = f.split('.')
                    split_2 = split_1[0].split('_')
                    bandwidth = int(split_2[-1])
                    len_buffer_edge = int(np.mean(pd.read_csv(my_path+f).values))
            except Exception:
                continue
        tmp = {"idx":idx, "sample_size":sample_size, "number_features":number_features, "budget":budget, "bandwidth":bandwidth, "len_buffer_edge":len_buffer_edge,
               "len_buffer_server":len_buffer_server, "rs":np.array(rs[:50]), "us":np.array(us[:50])}
        df = df.append(tmp, ignore_index=True)

    df.to_pickle('../outputs/'+file_name)

my_path = '../outputs/outputecg/'
file_name = "outputsynt.pickle"
extract_results(my_path, file_name)


def compare_results(file_path):
    model_size = np.load('../outputs/outputecg/model_size_256.npy')
    df = pd.read_pickle(file_path)
    df['nr_devices'] = np.tile(np.array([1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16]), 5)

    #df['nr_devices'] = np.tile(
    #   np.array([1,1,1,1,1,1,1,1,1]), 5)
    #df['nr_devices'] = np.tile(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), 1)
    #group = ["bandwidth"]
    group = ["budget"]
    #group = ["bandwidth", "budget"]
    #group = ["nr_devices"]
    sample_size = df.iloc[0].sample_size
    tmp_rs = df.groupby(by=group)['rs'].apply(list).reset_index()
    tmp_us = df.groupby(by=group)['us'].apply(list).reset_index()
    tmp_mean_edge = df.groupby(by=group)['len_buffer_edge'].mean().reset_index()
    tmp_mean_server = df.groupby(by=group)['len_buffer_server'].mean().reset_index()
    for i in range(len(tmp_rs)):
        tmp_rs.at[i, 'rs'] = np.array(tmp_rs.loc[i].rs).mean(axis=0).flatten()
        tmp_us.at[i, 'us'] = np.array(tmp_us.loc[i].us).mean(axis=0).flatten()

    tmp = pd.merge(tmp_rs, tmp_us, how='left', on=group)
    tmp = pd.merge(tmp, tmp_mean_edge, how='left', on=group)
    tmp = pd.merge(tmp, tmp_mean_server, how='left', on=group)
    tmp['model_size'] = model_size
    tmp['sample_size'] = sample_size
    plt.figure(figsize=(10, 7))
    col = ['k', 'r']
    line_style = ['--^', '--o', '--s', '-->']
    tmp.to_csv('../outputs/ecg_groupby'+str(group)+'.csv', index=False)
    for i in range(len(tmp)):

        plt.plot(100*tmp.iloc[i].rs, col[0]+line_style[i], label=str(group[0])+":" + str(tmp.iloc[i].budget) +
                                                             "; RS")
        plt.plot(100*tmp.iloc[i].us, col[1]+line_style[i], label=str(group[0])+":" + str(tmp.iloc[i].budget) +
                                                             "; US")

    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('number of labeled data points', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.tight_layout()
    plt.savefig('../outputs/ecg_groupby'+group[0]+'.pdf')
compare_results("../outputs/"+file_name)