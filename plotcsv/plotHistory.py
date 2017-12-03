from plotcsv import  plot_fn as pfn
import   matplotlib.pyplot as plt
data_keys = ['loss','val_loss']

cnnroot = '../logs/'
fcroot = '../logs/'

file_lists = {'CNN' : cnnroot+'cnn_history.csv',
              'FC' : fcroot+'fc_history.csv',
              }


d1 = pfn.plotHistory(file_lists,data_keys,
                     '../logs/')


plt.show()
