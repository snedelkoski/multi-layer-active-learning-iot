import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def to_float(arr):
    arr_new = []
    arr = list(arr.split(' '))
    for i in arr:
        try:
            arr_new.append(float(i.replace("[", '').replace("]", '').replace("\r\n", '')))
        except Exception:
            continue
    return arr_new


group = "labeling bugdet"
df_ecg = pd.read_csv('../outputs/paper_results/ecg_groupby[\'bandwidth\', \'budget\'].csv')

col = ['k', 'r']
line_style = ['--^', '--o', '--s', '-->']
print (df_ecg.us)
plt.figure(figsize=(10, 7))
for i in range(len(df_ecg)):
    us = to_float(df_ecg.iloc[i].us)
    rs = to_float(df_ecg.iloc[i].rs)
    plt.plot(rs, col[0]+line_style[i], label=str(group)+":" + str(df_ecg.iloc[i].budget) +
                                                             "s; RS")
    plt.plot(us, col[1]+line_style[i], label=str(group)+":" + str(df_ecg.iloc[i].budget) +
                                                             "s; US")
    plt.legend(fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('number of labeled data points', fontsize=13)
    plt.ylabel('accuracy', fontsize=13)
    plt.savefig('../outputs/synt_groupby'+group+'.pdf')