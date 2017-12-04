import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
def readData(file_lists, data_keys):
    '''
    read data from a list of files according keys.
    :param file_lists:
    :param data_keys:
    :return: dictionary storing the data
    '''
    data = {}
    for k in file_lists.keys():
        file_data = pd.read_csv(file_lists[k])
        data[k] = { m: file_data[m] for m in data_keys}
    return data


def plot(data,key,title,ylabel,xlabel,savepath):
    '''
    visiual the data
    :param data:
    :param key:
    :param title:
    :param ylabel:
    :param xlabel:
    :param savepath:
    :return:
    '''
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

    plt.show(block=False)



def plotHistory(file_lists,data_keys,savepth):
    data = readData(file_lists,data_keys)
    plot(data, 'loss', 'train loss', 'loss', 'epoch',savepth+'trainloss')
    plot(data, 'val_loss', 'val loss', 'loss', 'epoch', savepth + 'valloss')



