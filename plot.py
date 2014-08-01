import pandas as pd
import numpy as np
from sp2014a import savitzky_golay
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.ion()

sample = '0884'
path = "attenuation_corrected_{0}.p".format(sample)
data_df = pd.read_pickle(path)

plt.close('all')
# add vertical lines to plot where growth started/ended/temperature was changed
# time in seconds
growth_start = 60
growth_end = 5414
shutter_open = 5925
shutter_close = 6005
t_down = 6085
times = [growth_start, growth_end, shutter_open, shutter_close, t_down]
# final plot ctr-s
plt.figure('CTR')
plt.plot(data_df.time, data_df.ctr_center_mass_y,
         label='center of mass hk')
plt.plot(data_df.time, data_df.ctr_center_mass_x, label='center of mass l')
plt.xlabel('Time (s)')
plt.ylabel('Position (pixel)')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.savefig('ctr.png')
plt.show()

# final plot wz tw
plt.figure('WZ tW')
plt.plot(data_df.time, data_df.corr_data_tw, label='TW')
plt.plot(data_df.time, data_df.corr_data_wz, label='WZ')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('twwz.png')

# final plot fwhm
plt.figure('FWHM')
plt.plot(data_df.time, data_df.corr_data_fwhm_hk, label='FWHM hK')
plt.plot(data_df.time, data_df.corr_data_fwhm_l, label='FWHM l')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Width (pixel)')
plt.savefig('FWHM.png')

# final plot wz tw
plt.figure('WZ tW')
plt.plot(data_df.time, data_df.corr_data_tw, label='TW')
plt.plot(data_df.time, data_df.corr_data_wz, label='WZ')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('twwz.png')


def interpolate(values, time, no_points=1000):
    time = np.array(time)
    x = np.linspace(0, time.max() - 1, len(values))
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
#test interpolation before substraction

plt.figure('tW')
x,y = interpolate(data_df.corr_data_tw.values, data_df.time )
plt.plot(x,y, label='TW')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('twwz.png')

# final plot wz tw
plt.figure('tW')
plt.plot(data_df.time, savitzky_golay(
    data_df.corr_data_tw.values, 11, 2, 0), label='TW')
plt.plot(data_df.time, savitzky_golay(
    data_df.corr_data_wz.values, 11, 2, 0), label='TW')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('twwz.png')
# final plot wz tw
plt.figure('tW')
plt.plot(data_df.time, pd.rolling_mean(data_df.corr_data_tw.values,5) , label='TW')
plt.plot(data_df.time, savitzky_golay(
    data_df.corr_data_wz.values, 11, 2, 0), label='TW')
lines = [plt.axvline(x, c='k') for x in times]
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Intensity (a.u.)')
plt.savefig('twwz.png')