# imports
import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import time
from sp2014a import parse_img, parse_time
from scipy.ndimage.filters import median_filter
from scipy.stats.stats import nanmean
import matplotlib.pyplot as plt
plt.ion()

# names for sample and relative flatfield
name = str(sys.argv[1])
flat_field = str(sys.argv[2])

# data parsing
directory = './' + name + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i][-2:]

print '...start parsing'
time_Start = time.time()
data_array = np.array(map(parse_img, files))
data_array = data_array.astype(float)  # convert to float to use nans
times = np.array(map(parse_time, files))
data_array = data_array[np.argsort(times)]
# median 3x3 filter
data_array = median_filter(data_array, size=(1, 3, 3))

# flatfield parsing
directory = './' + flat_field + '/'
files = [directory + str(i) for i in os.listdir(directory) if 'tif' in i]

print '...start parsing flatfields'
flat_field_array = np.array(map(parse_img, files))
mean = np.mean(flat_field_array, axis=(0))
mean = mean.astype(float)
ff = mean / nanmean(mean)

# print out delta time
print 'parsed all files and flatfields.'
print 'time: {0} minute'.format((time.time() - time_Start) / 60)

data_array = data_array / ff
# x1,y1, x2,y2
polygon = [0, 60, 20, 100, 20, 100, 10, 30]
# width = ?
# height = ?

img = Image.fromarray(data_array[-1])
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = np.array(img)
plt.imshow(img)