from scipy.ndimage import gaussian_filter1d
from scipy import signal
import numpy as np
from derive4poiunt import *
import cPickle as pickle
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
import matplotlib
# # set up remote plotting
# remote = False
# if remote:
#     screen = ':0'
#     os.environ['DISPLAY'] = screen
#     print 'remote plotting on screen{}'.format(screen)

# ##########################
# ####### parsing files ###
# # #########################
# # #
name = '0884/time1'
flat_field = 'flatfield2/'
directory = './' + name + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i]
# # data parsing
# print '...start parsing/'
# time_Start = tm.time()
# data_array = np.array(map(parse_img_bkg, files))
# data_array = data_array.astype(float)  # convert to float to use nans
# data_array[data_array == 0.] = 1
times = np.array(map(parse_time, files))
# data_array = data_array[np.argsort(times)]
time = np.linspace(times.min(), times.max(), len(times))
# # data_array_bk = data_array
def load_a():
    '''
    load from  disk
    '''
    filename = 'array.p'
    infile = open(filename, 'r')
    arraty = pickle.load(infile)
    return arraty

time = time -time[0]
data_array_bk = load_a()
data_array_bk_mean = np.mean(rolling_window(data_array_bk, 7), -1)
data_array_bk_median = median_filter(data_array_bk_mean, size=(3, 3, 3))
diff_s = derive2d(time, data_array_bk_median)
diff = np.mean(rolling_window(diff_s, 10), -1)

sliced_arrays = diff[:,:,90:110] # mnaul slicing
sliced_arrays_k = diff[:,180:230,:] # manual slicing

l_sum_d_sliced_arrays = sliced_arrays.sum(axis=2)
k_sum_d_sliced_arrays = sliced_arrays_k.sum(axis=1)

wz_range_min = 1.42
wz_range_max  = 1.55
L = np.loadtxt('0884L.dat')
K = np.loadtxt('0884K.dat')
L = (L.mean(axis=0))
K = (K.transpose().mean(axis=0))
cond = (L > wz_range_min) & (L < wz_range_max)
L = L[cond]
select_WZ =  l_sum_d_sliced_arrays[:,cond]
peak_pos = []
peak_pos_L = []
peak_pos_i = []
delta = []
timez = tm.time()
L = L * 0.5976 # A-1
K = K * 1.69043 # A-1
for i in select_WZ:
    pks = signal.find_peaks_cwt(i,np.arange(0.3,1))
    if len(pks) > 1:
        peak_pos_L.append(L[pks])
        delta.append(L[pks][:-1]-L[pks][1:])
        peak_pos_i.append(i[pks])
        peak_pos.append(pks)
    else:
        peak_pos.append(np.NaN)


fig, ax = plt.subplots()
for values,timev in zip(peak_pos_i, time):
    values = np.array(values)
    ax.plot(np.ones(len(values)) *timev, values)
    # ax.plot(x= np.ones(len(values)) *timev, y = values)

plt.figure()
for d,values,timev in zip(peak_pos_L, peak_pos_i, time):
    values = np.array(values)
    d = d[values > tresh]
    plt.scatter(x= np.ones(len(d)) *timev, y =d)
plt.colorbar()


plt.figure()
for d,values,timev in zip(peak_pos_L, peak_pos_i, time):
    values = np.array(values)

    plt.scatter(x= np.ones(len(d)) *timev, y = d, c  = np.log(values), cmap=plt.cm.cubehelix)
plt.colorbar()

# plt.figure()
# for d,values,timev in zip(peak_pos_L, peak_pos_i, time):
#     values = arraty(values)
#     plt.scatter(x= np.ones(len(d)) *timev, y = values, c=d, cmap=plt.cm.rainbow)


sxl = []
dxl = []
ints = []
fuck = []
for d,values,timev in zip(peak_pos_L, peak_pos_i, time):
    values = np.array(values)
    center = np.where(values == values.max())[0]
    if values.max() > 9*np.exp(7):
        print values.max()
        try:
            sx =np.abs(d[center+1] -  d[center]  )
            dx =np.abs( d[center] - d[center-1])
            sxl.append(sx)
            dxl.append(dx)
            ints.append(values[center])
        except:
            sxl.append(np.array([np.nan]))
            dxl.append(np.array([np.nan]))
            continue
    else:
        sxl.append(np.array([np.nan]))
        dxl.append(np.array([np.nan]))

mean = np.mean(np.array([np.concatenate(dxl),np.concatenate(sxl)]), axis=0)
std = np.std(np.array([np.concatenate(dxl),np.concatenate(sxl)]), axis=0)
plt.errorbar(x=(time-time[0])[1:], y=(1/mean)/10, yerr=std)
plt.xlim(0,200)
plt.ylabel('Peak width (nm)')
plt.xlabel('Time (s)')

# fig, ax = plt.subplots()
# ax.set_yscale('log')
# b = np.sum(l_sum_d_sliced_arrays[200:205], axis=0)
# b = np.sign(b) * np.log(np.abs(b))
# ax.plot(L, b, label ='200')
# a = np.sum(l_sum_d_sliced_arrays[220:225], axis=0)
# a = np.sign(a) * np.log(np.abs(a))
# ax.plot(L,a, label ='220 ')
# c =np.sum(l_sum_d_sliced_arrays[230:235], axis=0)
# c = np.sign(c) * np.log(np.abs(c))
# ax.plot(L,c, label ='230 ')
# c =np.sum(l_sum_d_sliced_arrays[240:245], axis=0)
# c = np.sign(c) * np.log(np.abs(c))
# ax.plot(L,c, label ='240 ')
# c =np.sum(l_sum_d_sliced_arrays[250:255], axis=0)
# c = np.sign(c) * np.log(np.abs(c))
# ax.plot(L,c, label ='250')


fig2, ax = plt.subplots()
b = np.sum(l_sum_d_sliced_arrays[100:150], axis=0)
ax.plot(L, b, label ='100')
b = np.sum(l_sum_d_sliced_arrays[200:205], axis=0)
ax.plot(L, b, label ='200')
a = np.sum(l_sum_d_sliced_arrays[220:225], axis=0)
ax.plot(L,a, label ='220 ')
c =np.sum(l_sum_d_sliced_arrays[230:235], axis=0)
ax.plot(L,c, label ='230 ')
c =np.sum(l_sum_d_sliced_arrays[240:245], axis=0)
ax.plot(L,c, label ='240 ')
c =np.sum(l_sum_d_sliced_arrays[250:255], axis=0)
ax.plot(L,c, label ='250')

fig3, ax2 = plt.subplots()
shit = ax2.imshow(np.rot90(l_sum_d_sliced_arrays), interpolation=None, cmap=plt.cm.cubehelix, )
fig3.colorbar(shit)