#!/usr/bin/env python
# V1.0.0: Michele Cappellari, Oxford, 07 March 2011
# V1.0.1: Fixed deprecation warning. MC, Oxford, 1 October 2015
# V1.0.2: Changed imports for ltsfit as package. MC, Oxford, 17 April 2018    

import numpy as np
import matplotlib.pyplot as plt

from ltsfit.lts_linefit import lts_linefit

#------------------------------------------------------------------------------

def lts_linefit_example():
    """
    Usage example for lts_linefit()

    """
    ntot = 300 # Number of values

    # Coefficients of the test line
    #
    a = 10.
    b = 0.5

    n = int(ntot*0.6)  # 60% good values
    np.random.seed(913)
    x = np.random.uniform(18.5, 21.5, n)
    y = a + b*x

    sig_int = 0.3 # Intrinsic scatter in y
    y += np.random.normal(0, sig_int, n)

    sigx = 0.1 # Observational error in x
    sigy = 0.2 # Observational error in y
    x += np.random.normal(0, sigx, n)
    y += np.random.normal(0, sigy, n)

    # Outliers produce a background of spurious values
    # intersecting with the true distribution
    #
    nout = int(ntot*0.4)   # 40% outliers
    x1 = np.random.uniform((max(x) + min(x))/2, max(x), nout)
    y1 = np.random.uniform(20, 26, nout)

    # Combines the good values and the outliers in one vector
    #
    x = np.append(x, x1)
    y = np.append(y, y1)

    sigx = np.full_like(x, sigx)  # Adopted error in x
    sigy = np.full_like(x, sigy)  # Adopted error in y

    plt.clf()
    p = lts_linefit(x, y, sigx, sigy, pivot=np.median(x))
    plt.xlabel('x')
    plt.ylabel('$y, a + b\ (x - x_{0})$')
    plt.pause(1)

    # Illustrates how to obtain the best-fitting values from the class
    print("The best fitting parameters are:", p.ab)


#------------------------------------------------------------------------------

if __name__ == '__main__':

    lts_linefit_example()
