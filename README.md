# Equiv-FCL-CONVL

An equivalence of fully connected layer and convolutional layer.

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
  * trainnetworks.py        # train CNN and FC network.
  * visiualNet.py           # plot the architecture of the networks.
  * computeFnorm.py         # compare the two well tuned networks, plot historams of the weights and filters.
  * net.py                  # define CNN and FC network.
  * img2col.py              # converting 4D data to 2D matrix.
  * Data.py                 # data provider.
  * plotcsv.plotHistory.py  # plot the training and validation loss.
  * logger.BachLosses.py    # record the loss of every batch druing training.
```
## Running programs
```
    * train the two networks `python3 trainnetworks.py`. The log file and model are stored in the directory logs and model.
    * visiualize the two networks `python3 visualNet.py`. The reulsts are stored in the logs directory.
    * compare the two well-tuned networks, `python3 computeFnorm.py`.
    * visualize the losses of the two networks, `python3 plotHistory.py`.
```
## Authors

- [Wei Ma](https://github.com/Marvinmw)
- [Jun Lu](https://github.com/junlulocky)

## References

  1. Program [hipsternet](https://github.com/wiseodd/hipsternety)
  2. Andrea Vedaldi and Karel Lenc. Matconvnet: Convolutional neural networks for matlab. In Proceedings
     of the 23rd ACM international conference on Multimedia, pp. 689–692. ACM, 2015.

