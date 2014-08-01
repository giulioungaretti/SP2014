# imports
import h5py as hdf
from sp2014a import parse_img, parse_time
from scipy.ndimage.filters import median_filter
import os
import time
import numpy as np
import sys
from scipy.stats.stats import nanmean


# names for sample and relative flatfield
name = str(sys.argv[1])
flat_field = str(sys.argv[2])

# data parsing
directory = './'+name+'/'
files = [directory+str(i) for i in os.listdir(directory) if 'tif' in i]

print '...start parsing'
time_Start = time.time()
data_array = np.array(map(parse_img, files))
data_array = data_array.astype(float)  # convert to float to use nans
times = np.array(map(parse_time, files))
data_array = data_array[np.argsort(times)]
# median 3x3 filter
data_array = median_filter(data_array, size=(1, 3, 3))

# flatfield parsing
directory = './'+flat_field+'/'
files = [directory+str(i) for i in os.listdir(directory) if 'tif' in i]

print '...start parsing flatfields'
flat_field_array = np.array(map(parse_img, files))
mean = np.mean(flat_field_array, axis=(0))
mean = mean.astype(float)
ff = mean / nanmean(mean)

# print out delta time
print 'parsed all files and flatfields.'
print 'time: {0} minute'.format((time.time()-time_Start) / 60)

data_array = data_array / ff
# divide out background
bkg = data_array[:, 0:20, 0:20]

mean_bkg = np.ma.mean(bkg, axis=(1, 2))
data_array = data_array / mean_bkg[:, np.newaxis, np.newaxis]
data_array[data_array == 0.] = 1


# Save the data into a hdf5 file
filename = "data{0}.hdf5".format(name.replace('/', '_'))
# Create and open the file with the given name
outfile = hdf.File(filename)
outfile['data'] = data_array
# And close the file
outfile.close()
print 'dumped HDF5 file {0}'.format(filename)

# Save the data into a hdf5 file
filename = "data{0}{1}.hdf5".format(name.replace('/', '_'), '_time')
# Create and open the file with the given name
outfile = hdf.File(filename)
outfile['times'] = np.sort(times)
# And close the file
outfile.close()
print 'dumped HDF5 file {0}'.format(filename)
