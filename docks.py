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
from pyqtgraph.dockarea import *
import pyqtgraph.console
from derive4poiunt import derive2d, rolling_window

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

# names
name = str(sys.argv[1])
flat_field = str(sys.argv[2])
directory = './' + name + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i]
# data parsing
print '...start parsing/'
time_Start = tm.time()
data_array = np.array(map(parse_img_bkg, files))
data_array = data_array.astype(float)  # convert to float to use nans
data_array[data_array == 0.] = 1
times = np.array(map(parse_time, files))
data_array = data_array[np.argsort(times)]
time = np.linspace(times.min(), times.max(), len(times))

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
area = DockArea()
win.setCentralWidget(area)

######################
####### docks #######
######################
# Create docks, place them into the window one at a time.
# Note that size arguments are only a suggestion; docks will still have to
# fill the entire dock area and obey the limits of their internal widgets.
d_control = Dock("Control Panel", size=(100, 100))
d1 = Dock("Dock1", size=(500, 300))
d2 = Dock("Dock2", size=(500, 200))
d3 = Dock("Dock3", size=(500, 400))
d4 = Dock("Dock4", size=(500, 200))
d5 = Dock("Dock5", size=(500, 200))
d6 = Dock("Dock6", size=(500, 200))
d7 = Dock("Dock7", size=(500, 200))
d8 = Dock("Dock8", size=(500, 200))

area.addDock(d1, 'top')
area.addDock(d2, 'above', d1)  # place d2 at right edge of dock area
area.addDock(d3, 'above', d2)  # place d3 at bottom edge of d1
area.addDock(d4)  # place d4 at right edge of dock area
area.addDock(d5)  # place d5 at left edge of d1
area.addDock(d6, 'above', d5)  # place d5 at left edge of d1
area.addDock(d7)  # place d5 at left edge of d1
area.addDock(d8, 'above', d7)  # place d5 at left edge of d1
area.addDock(d_control, 'bottom')   # place d5 at top edge of d4
docks = [d1, d2, d4, d4, d5, d6, d7, d8]

######################
# main images display
#####################
dummy_plot1 = pg.PlotItem(title='Pilatus image')
dummy_plot2 = pg.PlotItem(title='Pilatus image')
dummy_plot3 = pg.PlotItem(title='Pilatus image')
dummy_plot4 = pg.PlotItem(title='Pilatus image')
dummy_plot5 = pg.PlotItem(title='Pilatus image')
dummy_plot6 = pg.PlotItem(title='Pilatus image')
dummy_plot7 = pg.PlotItem(title='Pilatus image')
dummy_plot8 = pg.PlotItem(title='Pilatus image')
imv1 = pg.ImageView(view=dummy_plot1)
imv2 = pg.ImageView(view=dummy_plot2)
imv3 = pg.ImageView(view=dummy_plot3)
imv4 = pg.ImageView(view=dummy_plot4)
imv5 = pg.ImageView(view=dummy_plot5)
imv6 = pg.ImageView(view=dummy_plot6)
imv7 = pg.ImageView(view=dummy_plot7)
imv8 = pg.ImageView(view=dummy_plot8)
image_widgets = [
    imv1,
    imv2,
    imv3,
    imv4,
    imv5,
    imv6,
    imv7,
    imv8
]

image_names = [
    'imv1',
    'imv2',
    'imv3',
    'imv4',
    'imv5',
    'imv6',
    'imv7',
    'imv8'
]

widgets_dict = {}
for name, obj in zip(image_names, image_widgets):
    widgets_dict[name] = obj

d1.addWidget(imv1)
d2.addWidget(imv2)
d3.addWidget(imv3)
d4.addWidget(imv4)
d5.addWidget(imv5)
d6.addWidget(imv6)
d7.addWidget(imv7)
d8.addWidget(imv8)

#########################
#########roi-bkg sub#####
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
    global rois, data_array_2
    global stack_0, stack_1, stack_2, imv3
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


########################
####### show data #######
########################
imv1.setImage(np.log(data_array_2), xvals=time)
dummy_plot1.setTitle('FF corrected Data')
imv2.setImage(np.log(data_array), xvals=time)
dummy_plot2.setTitle('Raw Data')

####### now correct for bkg  ########

def load_roi():
    global rois
    filename = 'roi_bkg.p'
    load_file = open(filename, 'r')
    roi_position = pickle.load(load_file)
    for roi, pos in zip(rois, roi_position):
        roi.setState(pos)

load_roi()
[roi.sigRegionChangeFinished.connect(update) for roi in rois]

######## show data ########
bkg = update()
data_array_bk = data_array_2 / (bkg[:, np.newaxis, np.newaxis])  # bkg corr
imv3.setImage(np.log(data_array_bk), xvals=time)
dummy_plot3.setTitle('Background corrected')
WZ_peak = pg.RectROI([170, 20], [170, 40], pen='y')
imv3.addItem(WZ_peak)


###########################
###### slicing ############
###########################
dummy_plot6.setTitle('slicing along l')
dummy_plot5.setTitle('slicing along k')
dummy_plot5.setLabels(bottom='time', left='K(pixel)')
dummy_plot6.setLabels(bottom='time', left='L(pixel)')


def do_k():
    Wz_slice = np.array([WZ_peak.getArrayRegion(
        data_array_bk[i],
        imv3.imageItem,
        axes=(0, 1)) for i in xrange(0, len(data_array))])
    WZ_sum_k = Wz_slice.sum(axis=2)
    imv6.setImage(np.log(WZ_sum_k), xvals=time)
    return WZ_sum_k



def do_L():
    Wz_slice = np.array([WZ_peak.getArrayRegion(
        data_array_bk[i],
        imv3.imageItem,
        axes=(0, 1)) for i in xrange(0, len(data_array))])
    WZ_sum_L = Wz_slice.sum(axis=1)
    imv5.setImage(np.log(WZ_sum_L), xvals=time)
    return WZ_sum_L

WZ_peak.sigRegionChangeFinished.connect(do_L)
WZ_peak.sigRegionChangeFinished.connect(do_k)

###########################
####### derivative  #######
###########################
# compute the mean over 3 images in time
data_array_bk_mean = np.mean(rolling_window(data_array_bk, 7), -1)
data_array_bk_median = median_filter(data_array_bk_mean, size=(3, 3, 3))
diff_s = derive2d(time, data_array_bk_median)
diff = np.mean(rolling_window(diff_s, 10), -1)
# median_filter(diff, size=(4, 3, 3))
dtime = time
imv4.setImage(diff, xvals=dtime)
dummy_plot4.setTitle('Derivative numerical')

dWZ_peak = pg.RectROI([170, 20], [170, 40], pen='y')
imv4.addItem(dWZ_peak)
###########################
###### slicing derivative##
###########################

dummy_plot7.setTitle('slicing along l')
dummy_plot8.setTitle('slicing along k')
dummy_plot8.setLabels(bottom='time', left='K(pixel)')
dummy_plot7.setLabels(bottom='time', left='L(pixel)')


def do_dk():
    Wz_slice = np.array([dWZ_peak.getArrayRegion(
        diff[i],
        imv4.imageItem,
        axes=(0, 1)) for i in xrange(0, len(data_array))])
    WZ_sum_k = Wz_slice.sum(axis=2)
    imv7.setImage(WZ_sum_k, xvals=dtime)
    return WZ_sum_k


def do_dL():
    Wz_slice = np.array([dWZ_peak.getArrayRegion(
        diff[i],
        imv4.imageItem,
        axes=(0, 1)) for i in xrange(0, len(data_array))])
    WZ_sum_L = Wz_slice.sum(axis=1)
    imv8.setImage(WZ_sum_L, xvals=dtime)
    return WZ_sum_L

dWZ_peak.sigRegionChangeFinished.connect(do_dL)
dWZ_peak.sigRegionChangeFinished.connect(do_dk)
win.show()
######################
# control panel
####################

w1 = pg.LayoutWidget()
label = QtGui.QLabel("Controls")
saveBtn = QtGui.QPushButton('Save dock state')
restoreBtn = QtGui.QPushButton('Restore dock state')
save_R = QtGui.QPushButton('Save Red ROIs')
r_roi = QtGui.QPushButton('Restore Red ROIs')
update_bk = QtGui.QPushButton('Update Background correction from Red ROIs')
combo = QtGui.QComboBox()
[combo.addItem(widget) for widget in image_names]
restoreBtn.setEnabled(False)
w1.addWidget(label, row=0, col=0)
w1.addWidget(saveBtn, row=1, col=0)
w1.addWidget(restoreBtn, row=2, col=0)
w1.addWidget(save_R, row=1, col=2)
w1.addWidget(r_roi, row=2, col=2)
w1.addWidget(update_bk, row=1, col=3)
w1.addWidget(combo, row=1, col=4)
d_control.addWidget(w1)
state = None


def save():
    global state
    state = area.saveState()
    restoreBtn.setEnabled(True)


def load():
    global state
    area.restoreState(state)


def dump_roi():
    global rois
    filename = 'roi_bkg.p'
    infile = open(filename, 'w')
    roi_pos = [roi.saveState() for roi in rois]
    pickle.dump(roi_pos, infile)
    infile.flush()


def update_bkg():
    global update, data_array_2, imv3
    bkg = update()
    data_array_bk = data_array_2 / (bkg[:, np.newaxis, np.newaxis])  # bkg corr
    imv3.setImage(np.log(data_array_bk), xvals=time)


def roi_select(text):
    global widgets_dict, imv
    plt.ion()
    plt.close('all')
    key = str(text)
    imv = widgets_dict[key]
    items_ = imv.getRoiPlot()
    curveitem = items_.items()[-14]
    fig, ax = plt.subplots(num='ROIplot')
    ax.plot(curveitem.getData()[0], curveitem.getData()[1], 'ko-')
    plt.show()
    return imv



def get_data(imv):
    items_ = imv.getRoiPlot()
    curveitem = items_.items()[-14]
    return curveitem.getData()[0], curveitem.getData()[1]


saveBtn.clicked.connect(save)
restoreBtn.clicked.connect(load)
save_R.clicked.connect(dump_roi)
r_roi.clicked.connect(load_roi)
update_bk.clicked.connect(update_bkg)
combo.activated[str].connect(roi_select)


def dump_array(array):
    '''
    dump array to disk
    '''
    filename = 'array.p'
    infile = open(filename, 'w')
    pickle.dump(array, infile)
    infile.flush()
