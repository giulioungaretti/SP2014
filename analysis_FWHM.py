import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import find


def FWHM(Y):
    try:
        np.shape(Y)[1]
        X = Y[0]
        Y = Y[1]
        # print 'x value loaded from'
    except:
        X = np.linspace(0, len(Y), len(Y))
    half_max = np.max(Y) / 1.8
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    # plt.plot(X[0:len(d)], d) #if you are interested
    #find the left and right most indexes
    try:
        left_idx = find(d > 0)[0]
        right_idx = find(d < 0)[-1]
        delta = X[right_idx] - X[left_idx]  # return the differenceo (full width)
    except Exception as e:
        delta = np.NaN
        print e
    return delta
