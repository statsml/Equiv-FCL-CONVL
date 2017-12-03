import plot_fn as pfn
import   matplotlib.pyplot as plt
data_keys = ['acc','loss','val_acc','val_loss']
#path = '/Users/mawei/Documents/tensormorph/plotcsv/logs/noise/11_0/history/'
root = '/Users/mawei/Documents/tensormorph/plotcsv/logs/original_nonnoise/'
#root = '/Users/mawei/Documents/tensormorph/plotcsv/logs/nonnoise/11_0/'
path = root
file_lists = {'scratch' : path+'history.csv',
              }


data_keys_train = ['acc','loss']
path = root
file_lists_train = {'scratch' : path+'train.csv',

              }

d1 = pfn.plotHistory(file_lists,data_keys,
                     path)
pfn.plottable(d1, data_keys,
              path+'val')

d2 = pfn.plotTrain(file_lists_train,data_keys_train,
                   path)
pfn.plottable(d2, data_keys_train,
              path+'train')


plt.show()
