# imports
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.misc import imshow
from time import mktime


class SPring8_image(object):
    '''
    Image class that reads from pilatus detector
    '''

    def __init__(self, file_name, size):
            """Init method for class Spring

            Parameters
            ----------
            file_name
            size is a list of coordinates to crop the image
            Returns
            -------
            None
            """
            self.file_name = file_name
            self.size = size

    def read_detector(self):
            '''
            Read the detector image and return a dictionary with the
            information contained
            in the header and the image as a numpy array.
            name can be 1 for ccd or 2 for pilatus
            TO - Do add angles from the diffractometers
            '''
            self.header_dict = {}
            byte = 4096
            in_file = open(self.file_name, 'rb')
            header_file = in_file.read(byte)
            self.image = in_file.read()  # read the rest of image
            in_file.close()  # save memory
            # perform median filter 5x5
            self.header = header_file.split('\n')
            self.time = self.header[0][30:49]

    def read_image(self):
            # convert the  binary image to integers
            res = [487, 195]  # rect
            user_dtype = np.uint32
            from_string = np.fromstring(self.image, dtype=user_dtype)
            self.img = from_string.reshape(res[1], res[0])
            self.img = self.img[self.size[0]:self.size[1],
                                self.size[2]:self.size[3]]
            # self.img = median_filter(self.img, 11)

    def read_img_bkg(self):
            # convert the  binary image to integers
            res = [487, 195]  # rect
            user_dtype = np.uint32
            from_string = np.fromstring(self.image, dtype=user_dtype)
            self.img_bkg = from_string.reshape(res[1], res[0])

    def show(self, log=False, size_fig=(10, 5)):
            'sowh detecotr image, optioanlly in log scale'
            fig, ax = plt.subplots(figsize=size_fig)
            ax.grid(color='white')
            if log:
                imshow(np.log(self.img), cmap=plt.cm.cubehelix)
            if not log:
                imshow(self.img, cmap=plt.cm.Oranges)
                plt.colorbar()
                return fig, ax


def parse_img(file_name, crop=[70, 135, 150, 370]):
    """extract image from spring8 object

    Parameters
    ----------
    file name of spring8 image file
    Returns
    -------
    numpy array of image rotated and corrected for a QT gui
    """
    img = SPring8_image(file_name, crop)
    img.read_detector()
    img.read_image()
    return np.rot90(np.flipud((img.img)), 3)


def parse_img_bkg(file_name, crop=[70, 135, 150, 370]):
    """extract full detector image.

    Parameters
    ----------
    file name of spring8 image file
    Returns
    ------
    numpy array of image rotated and corrected for a QT gui
    """
    img = SPring8_image(file_name, crop)
    img.read_detector()
    img.read_img_bkg()
    return np.rot90(np.flipud((img.img_bkg)), 3)


def parse_time(file_name, crop=[120, 125, 290, 370]):
    """extract time string from SPring8 object

    Parameters
    ----------
    file name of SPring 8 image file
    Returns
    -------
    timestamp of image
    """
    img = SPring8_image(file_name, crop)
    img.read_detector()
    time = datetime.strptime(img.time, '%Y:%m:%d %H:%M:%S')
    return (mktime(time.timetuple()))


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number. Symmetric window.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized symmetric window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2  # symmetric window

    # precompute coefficients
    b = np.mat([[k ** i for i in order_range]
               for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
