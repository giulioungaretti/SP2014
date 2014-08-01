import os
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
from numpy import sum, mean
import h5py as hdf
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
from collections import deque
from sp2014a import savitzky_golay
import sys
from scipy.ndimage.measurements import center_of_mass
import cPickle as pickle
from analysis_FWHM import FWHM
from scipy.interpolate import interp1d
import pandas as pd

try:
    name = sys.argv[1]
except:
    name = '0884'
filename = "data{0}.hdf5".format(name.replace('/', '_'))
infile = hdf.File(filename, "r")
data_array = np.array(infile['data'])


filename = "data{0}{1}.hdf5".format(name.replace('/', '_'), '_time')
infile = hdf.File(filename, "r")
time = np.array(infile['times'])
on_off = np.ones(1)

# set up remote plotting
remote = True
if remote:
    screen = ':0'
    os.environ['DISPLAY'] = screen
    print 'remote plotting on screen{}'.format(screen)

# qt GUI
app = pg.mkQApp()
# Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800, 1000)
win.setWindowTitle('DataSlicing sample:{0}'.format(name))
cw = QtGui.QWidget()
win.setCentralWidget(cw)

# layout
l = QtGui.QGridLayout()
cw.setLayout(l)

# main image display
dummy_plot = pg.PlotItem(title='Pilatus image @ Time:{0}'.format('0'))
dummy_plot.enableAutoRange()
imv1 = pg.ImageView(view=dummy_plot)
# create number of rois:
rois = [
    pg.RectROI([170, 20], [170, 40],
               pen='y'),
    pg.RectROI([250, 0], [250, 50],
               pen='r'),
    pg.RectROI([140, 20], [140, 40],
               pen='g'),  # WZ
    pg.RectROI([125, 20], [125, 40],
               pen=(255, 0, 123)),  # bkg left
    pg.RectROI([155, 20], [155, 40],
               pen=(255, 200, 123))  # bkg  right
]  # [x,y]

# get rois
roi1, roi2, roi3, roi4, roi5 = rois

# add rois to main image
[imv1.addItem(roi) for roi in rois]
# create number of plots
dummy_plots = [pg.PlotItem(title='roi image 1'),
               pg.PlotItem(title='roi image 2'),
               pg.PlotItem(title='roi image 3'),
               pg.PlotItem(title='roi image 4'),
               pg.PlotItem(title='roi image 5'), ]
# enable auto scale
[dummy_plot_i.enableAutoRange() for dummy_plot_i in dummy_plots]
# create the images using dummy as view
images = [pg.ImageView(view=i) for i in dummy_plots]

# unpack images widgets
imv2,  imv3,  imv4, imv5, imv6 = images
#  roi image display plots
plots = [pg.PlotWidget() for i in rois]
# unpack  plot widgets
pw, pw2, pw3, pw4, pw5 = plots
[dummy_plot_i.enableAutoRange() for dummy_plot_i in plots]

# plot for line cut in first row
pw6 = pg.PlotWidget()
pw6.enableAutoRange()
pw6.setLabel('left', "Intensity", units='a.u.')
pw6.setLabel('bottom', "position", units='pixel')
pw6.setLogMode(x=False, y=True)

# customize plots
for i in plots:
    i.setLabel('left', "Intensity", units='a.u.')
    i.setLabel('bottom', "Time", units='s')
    i.setLogMode(x=False, y=False)
    i.enableAutoRange()

# add widgets
# fromRow, int fromColumn, int rowSpan, int columnSpan
l.addWidget(imv1, 1, 0, 1, 1)
l.addWidget(pw, 2, 0, 1, 1)
l.addWidget(pw2, 3, 0, 1, 1)
l.addWidget(pw3, 4, 0, 1, 1)

# add text to main widget to see the timestamp!


# define update text function
def update_index_text():
    global dummy_plot, imv1
    title = 'Pilatus image @ Time:{0}'.format(time[imv1.currentIndex])
    dummy_plot.setTitle(title)
    return title


# define update  image function for  each roi
def update_1():
    d4 = roi1.getArrayRegion(
        data_array[imv1.currentIndex], imv1.imageItem, axes=(0, 1))
    imv2.setImage(d4)


def update_2():
    d4 = roi2.getArrayRegion(
        data_array[imv1.currentIndex], imv1.imageItem, axes=(0, 1))
    imv3.setImage(d4)


def update_3():
    d4 = roi3.getArrayRegion(
        data_array[imv1.currentIndex], imv1.imageItem, axes=(0, 1))
    imv4.setImage(d4)


def update_4():
    d4 = roi4.getArrayRegion(
        data_array[imv1.currentIndex], imv1.imageItem, axes=(0, 1))
    imv5.setImage(d4)


def update_5():
    d4 = roi5.getArrayRegion(
        data_array[imv1.currentIndex], imv1.imageItem, axes=(0, 1))
    imv6.setImage(d4)

derive = False


def update_sum_plot_1():
    'yellow roi - ctr '
    global data_array, imv1, plots, roi1, plot1
    datas = deque()
    for i in xrange(0, len(data_array)):
        datas.append(center_of_mass(roi1.getArrayRegion(data_array[i],
                                                        imv1.imageItem, axes=(0, 1))))
    if derive == True:
        plot1.plot(
            time, savitzky_golay(list(datas), 15, 2, 1), clear=True, pen=None, symbol='o',
            symbolPen=None, symbolSize=3, symbolBrush=roi1.pen.color())
    else:
        plot1.plot(time, np.array(datas)[:, 0], clear=True, pen=None, symbol='o',
                   symbolPen=None, symbolSize=3, symbolBrush=roi1.pen.color())
        plot1.plot(time, np.array(datas)[:, 1], pen=None, symbol='s',
                   symbolPen=None, symbolSize=3, symbolBrush=roi1.pen.color())
    plot4.setLabel('top', "Center OF mass yellow roi", units=None)
    return np.array(datas)


def update_sum_plot_2():
    'red roi'
    global data_array, imv1, plots, roi2, plot2
    datas = deque()
    for i in xrange(0, len(data_array)):
        datas.append(sum(roi2.getArrayRegion(data_array[i],
                                             imv1.imageItem, axes=(0, 1))))
    # datas =  np.sum(datas, axis=1)
    if derive == True:
        plot2.plot(
            time, savitzky_golay(list(datas), 15, 2, 1), clear=True, pen=None, symbol='o',
            symbolPen=None, symbolSize=3, symbolBrush=roi2.pen.color())
    else:
        plot2.plot(time, list(datas), clear=True, pen=None, symbol='o',
                   symbolPen=None, symbolSize=3, symbolBrush=roi2.pen.color())
    plot4.setLabel('top', "Red ROI", units=None)
    return np.array(datas)


def update_sum_plot_3():
    "green roi WZ"
    global data_array, imv1, plots, roi3, plot3
    datas = deque()
    for i in xrange(0, len(data_array)):
        datas.append(sum(roi3.getArrayRegion(data_array[i],
                                             imv1.imageItem, axes=(0, 1))))
    # datas =  np.sum(datas, axis=1)
    if derive == True:
        plot3.plot(
            time, savitzky_golay(list(datas), 15, 2, 1), clear=True, pen=None, symbol='o',
            symbolPen=None, symbolSize=3, symbolBrush=roi3.pen.color())
    else:
        plot3.plot(time, list(datas), clear=True, pen=None, symbol='o',
                   symbolPen=None, symbolSize=3, symbolBrush=roi3.pen.color())
    plot3.setLabel('top', "Green ROI", units=None)
    return np.array(datas)


def update_sum_plot_4():
    "magenta roi  TW"
    global data_array, imv1, plots, roi4, plot4
    datas = deque()
    for i in xrange(0, len(data_array)):
        datas.append((roi4.getArrayRegion(data_array[i],
                                          imv1.imageItem, axes=(0, 1))))
    return np.array(datas)


def update_sum_plot_5():
    "orange roi "
    global data_array, imv1, plots, roi5, plot5
    datas = deque()
    for i in xrange(0, len(data_array)):
        datas.append(mean(roi5.getArrayRegion(data_array[i],
                                              imv1.imageItem, axes=(0, 1))))


update_plots = [update_sum_plot_1, update_sum_plot_2,
                update_sum_plot_5, update_sum_plot_3, update_sum_plot_4]

plot1, plot2, plot3, plot4, plot5 = plots
# link time axis
for p in plots:
    p.setXLink(plots[0])

roi1.sigRegionChangeFinished.connect(update_sum_plot_1)
roi2.sigRegionChangeFinished.connect(update_sum_plot_2)
roi3.sigRegionChangeFinished.connect(update_sum_plot_3)
roi4.sigRegionChangeFinished.connect(update_sum_plot_4)
roi5.sigRegionChangeFinished.connect(update_sum_plot_5)

imv1.sigTimeChanged.connect(update_index_text)
imv1.sigTimeChanged.connect(update_1)
imv1.sigTimeChanged.connect(update_2)
imv1.sigTimeChanged.connect(update_3)
imv1.sigTimeChanged.connect(update_4)
imv1.sigTimeChanged.connect(update_5)

# intialized the image
imv1.setImage(np.log(data_array), xvals=time)

win.show()


def dump_roi():
    filename = 'roi_position'
    infile = open(filename, 'w')
    roi_pos = [roi.saveState() for roi in rois]
    pickle.dump(roi_pos, infile)
    infile.flush()


def load_roi():
    filename = 'roi_position'
    load_file = open(filename, 'r')
    roi_position = pickle.load(load_file)
    for roi, pos in zip(rois, roi_position):
        roi.setState(pos)


def dump_sum4(plot=False):
    wz_array = update_sum_plot_4()
    np.shape(wz_array)
    sum_along_L = np.sum(wz_array, axis=2)
    np.shape(sum_along_L)
    sum_along_hk = np.sum(wz_array, axis=1)
    if plot:
        plt.figure()
        for i in xrange(0, len(data_array)):
            plt.plot(sum_along_L[i], 'o')
        # plt.figure()
        for i in xrange(0, len(data_array)):
            plt.plot(sum_along_hk[i], 'o')
    return sum_along_hk, sum_along_L


def interpolate(values, no_points=1000):
    x = np.linspace(0, len(values) - 1, len(values))
    f = interp1d(x, values, kind='cubic')
    x_new = np.linspace(0, len(values) - 1, no_points)
    y_new = f(x_new)
    return x_new, y_new


def interpolate_arrays(arrays, plot=False, no_points=1000):
    interpolaed_arrays = np.array(map(interpolate, arrays))
    if plot:
        for i in xrange(0, len(interpolaed_arrays)):
            plt.plot(interpolaed_arrays[0][0], interpolaed_arrays[i][1], '-')
    return interpolaed_arrays


def do(sum_along_hk, sum_along_L, plot=False):
    FWHMs_L = np.array(map(FWHM, sum_along_L))
    FWHMs_hk = np.array(map(FWHM, sum_along_hk))
    if plot:
        plt.figure()
        plt.plot(time, FWHMs_hk, 'o', label='FWHM hk')
        plt.plot(time, FWHMs_L, 'o', label='FWHM L')
        plt.legend()
    return FWHMs_hk, FWHMs_L


def plot_collection_raw(array, style='o', step=1):
    for i in xrange(0, len(array), step):
        plt.plot(array[i], style)


def plot_collection_intpr(array, style='-', step=1):
    timez = time - time[0]
    for i, t in zip(xrange(0, len(array), step), timez[::step]):
        plt.plot(array[0][0], array[i][1], style,
                 label='Time= {0}'.format(t))

load_roi()

# create dataframe to store the files
time_data = pd.DataFrame(index=time)

ctr_center_mass = np.array(update_sum_plot_1())

ctr_center_mass_x = ctr_center_mass[:, 0]
time_data['ctr_center_mass_x'] = ctr_center_mass_x
ctr_center_mass_y = ctr_center_mass[:, 1]
time_data['ctr_center_mass_y'] = ctr_center_mass_y
tw_intensity = np.array(update_sum_plot_2())
time_data['tw_intensity'] = tw_intensity
wz_intensity = np.array(update_sum_plot_3())
time_data['wz_intensity'] = wz_intensity

sum_along_hk, sum_along_L = dump_sum4()
time_data['sum_along_hk'] = sum_along_hk.max()
time_data['sum_along_L'] = sum_along_L.max()
intrp_hk = interpolate_arrays(sum_along_hk)
intrp_L = interpolate_arrays(sum_along_L)
no_interp_fwhm = do(sum_along_hk, sum_along_L)
interp_fwhm = do(intrp_hk, intrp_L)

time_data['FWHM_hk_raw'] = no_interp_fwhm[0]
time_data['FWHM_l_raw'] = no_interp_fwhm[1]
time_data['FWHM_hk'] = interp_fwhm[0]
time_data['FWHM_l'] = interp_fwhm[1]
time_data.to_pickle("results_{0}.p".format(name.replace('/', '_')))
