from src.plotcsv import  plot_fn as pfn
import   matplotlib.pyplot as plt
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
data_keys = ['loss','val_loss']
root = d+'/logs/adam/logs/'
file_lists = {'CNN' : root+'cnn_history.csv','FC' : root+'fc_history.csv'}
d1 = pfn.plotHistory(file_lists, data_keys, root)

root = d+'/logs/sgd/logs/'
file_lists = {'CNN' : root+'cnn_history.csv', 'FC' : root+'fc_history.csv'}
pfn.plotHistory(file_lists,data_keys, root)
plt.show()
