import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def FWHM(X,Y):
    half_max = np.max(Y) / 5
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
    return delta, right_idx, left_idx

#start with last value so peak is there
y_real = intrp_L[-1][1]
x = intrp_L[val][0]
max_pos = x[y_real==y_real.max()]
widht, pos1, pos2 =  FWHM(x,y)
plt.plot(x,y_real)
plt.axvline(x[pos1])
plt.axvline(x[pos2])

#take away the peak and spline fit
y_bkg = np.concatenate([y_real[:pos1], y_real[pos2:]])
x_bkg = np.concatenate([x[:pos1], x[pos2:]])
f = interp1d(x, y_real, kind='cubic')
plt.plot(x_bkg, y_bkg)
plt.plot(x,f(x))

plt.close()

