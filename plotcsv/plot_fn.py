import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
from scipy.signal import savgol_filter
def readData(file_lists, data_keys):
    data = {}
    for k in file_lists.keys():
        file_data = pd.read_csv(file_lists[k])
        data[k] = { m: file_data[m] for m in data_keys}
    return data


def plot(data,key,title,ylabel,xlabel,savepath,max_itr=1000,wisize=9):
    plt.figure()
    for k in data.keys():
         y=data[k][key][:]
         plt.plot(np.arange(len(y)),y,label=k)
    plt.title(title,fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.xlabel(xlabel,fontsize = 15)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if 'acc' in key:
        plt.legend(loc='lower right')
        if 'val' in key:
            plt.ylim(0.7,0.85)
        else:
            plt.ylim(0.7, 0.87)
    else:
        plt.legend(loc='upper right')
    suffix = ['.jpg', '.pdf']
    plt.gca().grid(True)
    for name in suffix:
        plt.savefig(savepath+name)
    #plt.ylim(0.5, 1)

    plt.show(block=False)



def plotHistory(file_lists,data_keys,savepth):
    data = readData(file_lists,data_keys)
    plot(data, 'loss', 'train loss', 'loss', 'epoch',savepth+'trainloss')
    plot(data, 'val_loss', 'val loss', 'loss', 'epoch', savepth + 'valloss')



