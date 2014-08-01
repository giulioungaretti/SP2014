import os
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import h5py as hdf
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
import sys
import cPickle as pickle
import time as tm
from sp2014a import parse_img, parse_time, parse_img_bkg
from scipy.stats.stats import nanmean
from collections import deque

# set up remote plotting
remote = False
if remote:
    screen = ':0'
    os.environ['DISPLAY'] = screen
    print 'remote plotting on screen{}'.format(screen)

##########################
####### parsing files ###
#########################

try:
    name = sys.argv[1]
except:
    name = '0884'

# load time
filename = "data{0}{1}.hdf5".format(name.replace('/', '_'), '_time')
infile = hdf.File(filename, "r")
time = np.array(infile['times'])
# names
name = str(sys.argv[1])
flat_field = str(sys.argv[2])
directory = './' + name + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i]
# data parsing
print '...start parsing'
time_Start = tm.time()
data_array = np.array(map(parse_img_bkg, files))
data_array = data_array.astype(float)  # convert to float to use nans
data_array[data_array == 0.] = 1
times = np.array(map(parse_time, files))
data_array = data_array[np.argsort(times)]

# names
directory = './' + flat_field + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i]
# flatfield parsing
print '...start parsing flatfields'
flat_field_array = np.array(map(parse_img_bkg, files))
flat_field_array = flat_field_array.astype(float)
flat_field_array[flat_field_array == 0.] = 1

# create mean flatfield
mean = np.mean(flat_field_array, axis=(0))
mean = mean.astype(float)
ff = mean / np.mean(mean)

print 'parsed all files and flatfields.'
print 'time: {0} minute'.format((tm.time() - time_Start) / 60)

####### correct for flatfield ########
data_array_2 = data_array / ff

##########################
###### qt GUI ###########
#########################


app = pg.mkQApp()
# Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800, 1000)
cw = QtGui.QWidget()
win.setCentralWidget(cw)

# layout
l = QtGui.QGridLayout()
cw.setLayout(l)

######################
# main images display
#####################
dummy_plot1 = pg.PlotItem(title='Pilatus image')
dummy_plot2 = pg.PlotItem(title='Pilatus image')
dummy_plot3 = pg.PlotItem(title='Pilatus image')
imv1 = pg.ImageView(view=dummy_plot1)
imv2 = pg.ImageView(view=dummy_plot2)
imv3 = pg.ImageView(view=dummy_plot3)
l.addWidget(imv1, 1, 0, 1, 1)
l.addWidget(imv2, 2, 0, 1, 1)
l.addWidget(imv3, 3, 0, 1, 1)

######### roi ###########
#########################
roi_0 = pg.RectROI([170, 20], [170, 40], pen='r')
roi_1 = pg.RectROI([170, 20], [170, 40], pen='r')
roi_z = pg.RectROI([170, 20], [170, 40], pen='r')
imv1.addItem(roi_0)
imv1.addItem(roi_1)
imv1.addItem(roi_z)
global rois
rois = [roi_1, roi_0, roi_z]


def update():
    """add 3 square red rois on imv1 item (top plot)
    return the background signal
    """
    global rois
    global stack_0, stack_1, stack_2
    roi_0, roi_1, roi_z = rois
    stack_0 = np.array([roi_0.getArrayRegion(
        data_array_2[i], imv1.imageItem, axes=(0, 1)) for i in xrange(0, len(data_array))])
    stack_1 = np.array([roi_1.getArrayRegion(
        data_array_2[i], imv1.imageItem, axes=(0, 1)) for i in xrange(0, len(data_array))])
    stack_2 = np.array([roi_z.getArrayRegion(
        data_array_2[i], imv1.imageItem, axes=(0, 1)) for i in xrange(0, len(data_array))])
    bkg_0 = stack_0.mean(axis=(1, 2))
    bkg_1 = stack_1.mean(axis=(1, 2))
    bkg_2 = stack_2.mean(axis=(1, 2))
    bkg = (bkg_0 + bkg_1 + bkg_2) / 3  # manual mean
    bkg = bkg / bkg.mean()
    return bkg


def dump_roi():
    global rois
    filename = 'roi_bkg.p'
    infile = open(filename, 'w')
    roi_pos = [roi.saveState() for roi in rois]
    pickle.dump(roi_pos, infile)
    infile.flush()


def load_roi():
    global rois
    filename = 'roi_bkg.p'
    load_file = open(filename, 'r')
    roi_position = pickle.load(load_file)
    for roi, pos in zip(rois, roi_position):
        roi.setState(pos)



########################
####### show data #######
########################

imv1.setImage(np.log(data_array_2), xvals=time)
dummy_plot1.setTitle('FF corrected Data')
imv2.setImage(np.log(data_array), xvals=time)
dummy_plot2.setTitle('Raw Data')

####### now correct for bkg  ########

load_roi()
[roi.sigRegionChangeFinished.connect(update) for roi in rois]
meaned = update()  # very low intensity, as it should
data_array_bk = data_array_2 / (meaned[:, np.newaxis, np.newaxis])  # bkg corr

######## show data ########

imv3.setImage(np.log(data_array_bk), xvals=time)
dummy_plot3.setTitle('Background corrected')
win.show()

###########################
####### derivative  #######
###########################

diff = np.diff(data_array_bk)
imv2.setImage(diff, xvals=time)
dummy_plot2.setTitle('Derivative numerical')

###########################
###### slicing ############
###########################

WZ_peak = pg.RectROI([170, 20], [170, 40], pen='y')
imv3.addItem(WZ_peak)



def do():
    Wz_h_slice = np.array([WZ_peak.getArrayRegion(
        data_array_bk[i], imv3.imageItem, axes=(0, 1)) for i in xrange(0, len(data_array))])
    WZ_sum_L = Wz_h_slice.sum(axis=1)
    imv2.setImage(WZ_sum_L, xvals=time)

WZ_peak.sigRegionChangeFinished.connect(do)

# get out the array vs tiem
# imv2.setImage(np.diff(WZ_sum_L), xvals = time)


def plot_roi(imv):
    items_ = imv.getRoiPlot()
    curveitem = items_.items()[-14]
    print curveitem.getData()
    fig, ax = plt.subplots()
    ax.plot(curveitem.getData()[0], curveitem.getData()[1], 'ko-')
    return fig
