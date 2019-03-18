# ZeLiC - A novel interpolation technique

## Description

ZeLiC is an interpolation algorithm that can be used to interpolate data that has been sampled using with Lebesgue sampling. Lebesgue sampling consists of choosing a condition to sample one point of a signal. For instance, one condition could be : take one point when the signal is larger than a given threshold.

This Git contains 3 files:
- *recon_algs.py* contains many algorithms that can be used for interpolation and signal reconstruction
- *recon_utils.py* contains utilities that are used for the ZeLiC interpolation
-  *zeli_algorithm.py* contains the code for ZeLiC interpolation

### Libraries needed
For the *zeli_algorithm* script to work correctly, the user must have:
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/) 
There is an import of  the [fastdtw](https://pypi.org/project/fastdtw/) library, but it can be commented out as it is not crucial for the algorithm to work.

For the *recon_algs* script to work correctly, the user must have:
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [cvxpy](https://www.cvxpy.org/)
- [spams](http://spams-devel.gforge.inria.fr/)
## Authors

This algorithm has been made by Matthieu Bellucci, with the help of Luis Miralles and Atif Qureshi.
