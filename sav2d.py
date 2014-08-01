def derive2d(time, y):
    '''
    calculate dy by 4-point center differencing using array slices

    \frac{y[i-2] - 8y[i-1] + 8[i+1] - y[i+2]}{12h}

    y[0] and y[1] must be defined by lower order methods
    and y[-1] and y[-2] must be defined by lower order methods
    '''
    import numpy as np
    from scipy.ndimage.filters import median_filter
    time = time - time[0]
    x = np.linspace(time.min(), time.max(), len(time))
    dy = np.zeros(y.shape, np.float)  # we know it will be this size
    h = x[1] - x[0]  # this assumes the points are evenely spaced!
    dy[2:-2] = (y[0:-4] - 8 * y[1:-3] + 8 * y[3:-1] - y[4:]) / (12.0 * h)
    # simple differences at the end-points
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    dy[1] = (y[2] - y[1]) / (x[2] - x[1])
    dy[-2] = (y[-2] - y[-3]) / (x[-2] - x[-3])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    dy = median_filter(dy, size=(1, 3, 3))
    return dy
