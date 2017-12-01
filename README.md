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
```
  
## Python files

```
  * equivalence.py    # train the networks, type `python equivalence.py` in the terminal
  * cifar.py          # define the networks
  * img2col.py        # converting convolutional operation to matrix multplication
  * train.py          # optimization
```

## Other things

- please make storing log directory by `mkdir logs` before running equivalence.py
- please make storing model directory by `mkdir model` before running equivalence.py
  
## Authors

- [Wei Ma](https://github.com/Marvinmw)
- [Jun Lu](https://github.com/junlulocky)

## References


  1. [hipsternet](https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py)
  2. Andrea Vedaldi and Karel Lenc. Matconvnet: Convolutional neural networks for matlab. In Proceedings
     of the 23rd ACM international conference on Multimedia, pp. 689–692. ACM, 2015.

