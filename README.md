# Equiv-FCL-CONVL

An equivalence of fully connected layer and convolutional layer.
<<<<<<< HEAD

=======
Before running the program, please execute:
```
 * 'mkdir src/logs'
 * 'mkdir src/model'
```
>>>>>>> 1044d31bb886ef93b270e71ceef6f43c260adccf
## Dependence packages

```
  * numpy
  * python3
  * tensorflow
  * keras
  * panda
  * h5py
  * matplotlib
  * skimage
```
  
## Python files

```
  * equivalence.py          # test the equivalence of convolutional operation and matrix multplication, type `python equivalence.py` in the terminal
  * net.py                  # define CNN network
  * img2col.py              # converting 4D data to 2D matrix
  * Data.py                 # data provider
  * trainnetworks.py        # train CNN and FC network
  * visiualNet.py           # plot the architecture of the networks
  * computeFnorm.py         # compute F-norm of the outputs of the CONV layer and the dense layers, plot historams of the wights and filters
  * plotcsv.plotHistory     # plot the training and validation loss
  * logger.BachLosses.py    # record the loss of every batch
```

## Authors

- [Wei Ma](https://github.com/Marvinmw)
- [Jun Lu](https://github.com/junlulocky)

## References


  1. Program [hipsternet](https://github.com/wiseodd/hipsternety)
  2. Andrea Vedaldi and Karel Lenc. Matconvnet: Convolutional neural networks for matlab. In Proceedings
     of the 23rd ACM international conference on Multimedia, pp. 689–692. ACM, 2015.

