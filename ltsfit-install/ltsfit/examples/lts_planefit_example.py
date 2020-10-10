#!/usr/bin/env python
# V1.0.0: Michele Cappellari, Oxford, 5 November 2014
# V1.0.1: Fixed deprecation warning. MC, Oxford, 1 October 2015
# V1.0.2: Changed imports for ltsfit as package. MC, Oxford, 17 April 2018    

import numpy as np
import matplotlib.pyplot as plt

from ltsfit.lts_planefit import lts_planefit

#------------------------------------------------------------------------------


def lts_planefit_example():
    """
    Usage example for lts_planefit()

    """
    ntot = 300  # Number of values

    # Coefficients of the test line
    #
    a = 10.
    b = 2.
    c = 1.

    n = int(ntot*0.6)
    np.random.seed(915)
    x = np.random.uniform(17.5, 22.5, n)
    y = np.random.uniform(7.5, 12.5, n)
    z = a + b*x + c*y

    sig_int = 1  # Intrinsic scatter in z
    z = np.random.normal(z, sig_int, n)

    sigx = 0.2  # Observational error in x
    sigy = 0.4  # Observational error in y
    sigz = 0.4  # Observational error in z
    x = np.random.normal(x, sigx, n)
    y = np.random.normal(y, sigy, n)
    z = np.random.normal(z, sigz, n)

    # Outliers produce a background of spurious values
    # intersecting with the true distribution
    #
    nout = int(ntot*0.4)  # 40% outliers
    x1 = np.random.uniform(10, 30, nout)
    y1 = np.random.uniform(20, 40, nout)
    z1 = np.random.uniform((max(z) + min(z))/2, max(z), nout)

    # Combines the good values and the outliers in one vector
    #
    x = np.append(x, x1)
    y = np.append(y, y1)
    z = np.append(z, z1)

    sigx = np.full_like(x, sigx)  # Adopted error in x
    sigy = np.full_like(x, sigy)  # Adopted error in y
    sigz = np.full_like(x, sigz)  # Adopted error in z

    plt.clf()
    p = lts_planefit(x, y, z, sigx, sigy, sigz, pivotx=np.median(x), pivoty=np.median(y))
    plt.xlabel('z')
    plt.ylabel('$a + b\ (x - x_{0}) + c\ (y - y_{0})$')
    plt.pause(1)

    # Illustrates how to obtain the best-fitting values from the class
    print("The best fitting parameters are:", p.abc)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    lts_planefit_example()
