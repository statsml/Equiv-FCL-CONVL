from __future__ import print_function
import scipy.misc
import os
import cifar
import numpy as np
import train as tr
print('Prepare Data \n')

(x_train, y_train), (x_test, y_test), input_shape, batch_size, num_classes, epoches =  tr.getMiniData()
x=x_train
y=y_train


names  = [27,28]
directory ='./logs/'
d=[]
for i in names:
    directory = './logs/'+str(i)+'_cnn'+'/'
    d.append(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

strides = [2,      1 ]
kernels = [(27,27),(28,28)]
paddings =['valid','valid']
ch       = [28*28,  28*28]

for j in range(len(names)):
    print(names[j])
    cnneq = cifar.cnnOneLayer(input_shape, strides[j], kernels[j], ch[j], padding=paddings[j])
    print(d[j]+str(names[j])+'.csv')
    cnneq, h1 = tr.trainMSE(cnneq, x, x, x_test, x_test, d[j]+str(names[j])+'.csv', epoches=epoches, batch_size=batch_size)


    cnnimgs = cnneq.predict(x_train[200:210],batch_size=1,verbose=0)
    cnnimgs=np.reshape(cnnimgs,cnnimgs.shape[:3])
    for m in range(10):
        scipy.misc.imsave(d[j]+'cnnimg'+str(m)+'.png',cnnimgs[m,:,:])

if not os.path.exists('./logs/fc/'):
        os.makedirs('./logs/fc/')

fceq  = cifar.fcOneLayer(input_shape)
fceq, h2 = tr.trainMSE(fceq, x, x, x_test, x_test,'./logs/fc/fc.csv', epoches=epoches, batch_size=batch_size)

fcimgs = fceq.predict(x_train[200:210],batch_size=1,verbose=0)
fcimgs = np.reshape(fcimgs,fcimgs.shape[:3])
for k in range(10):
    scipy.misc.imsave('./logs/fc/fcimg' + str(k) + '.png', fcimgs[k, :, :])
