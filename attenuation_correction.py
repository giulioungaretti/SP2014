from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# set up remote plotting
remote = True
if remote:
    screen = ':0'
    os.environ['DISPLAY'] = screen
    print 'remote plotting on screen{}'.format(screen)

# load dataframe with un-corrected data
files = [str(i) for i in os.listdir('.') if 'results' in i]
data_frames = [pd.read_pickle(i) for i in files]
for df in data_frames:
    df['time'] = np.linspace(
        df.index.values.min(), df.index.values.max(), len(df.index))
data = pd.concat(data_frames)
data = data.sort(['time'])


# data = pd.concat(data_frames)
# data = data.sort_index()

# FWHM
plt.close()
# time = data.index.values -data.index.values[0]
time =  data.time.values
time = time - time[0]
corr_data = data.FWHM_l.values.copy()
plt.plot(time, data.FWHM_l.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10500
b = 10502
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
corr_data[b - 2] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)

corr_data_fwhm_l = corr_data

plt.close()
corr_data = data.FWHM_hk.values.copy()
plt.plot(time, data.FWHM_hk.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10500
b = 10502
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
corr_data[b - 2] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)

corr_data_fwhm_hk = corr_data

# tw and wz
plt.close()
corr_data = data.tw_intensity.values.copy()
plt.plot(time, data.tw_intensity.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10500
b = 10501
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)
corr_data_tw = corr_data

plt.close()
corr_data = data.wz_intensity.values.copy()
plt.plot(time, data.wz_intensity.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10499
b = 10501
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)
corr_data_wz = corr_data

# ctr center of mass
plt.close()
corr_data = data.ctr_center_mass_x.values.copy()
plt.plot(time, data.ctr_center_mass_x.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10500
b = 10501
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)
ctr_center_mass_x = corr_data

plt.close()
corr_data = data.ctr_center_mass_y.values.copy()
plt.plot(time, data.ctr_center_mass_y.values)
a = 1080
b = 1083
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)

a = 3464
b = 3466
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 10500
b = 10501
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
corr_data[b - 1] = np.nan
data_corr_1 = corr_data[b:] * norm_1
corr_data[b:] = data_corr_1
plt.plot(time, corr_data)

a = 11745
b = 11748
plt.plot(time[a], corr_data[a], 'ko')
plt.plot(time[b], corr_data[b], 'ko')
norm_1 = corr_data[a] / corr_data[b]
print norm_1
data_corr_1 = corr_data[b:] * norm_1
corr_data[b - 2] = np.nan
corr_data[b - 1] = np.nan
corr_data[b:] = data_corr_1

plt.plot(time, corr_data)
plt.close()
plt.plot(time, corr_data)
ctr_center_mass_y = corr_data

# pandifiy
import pandas as pd
data_df = pd.DataFrame(data={'ctr_center_mass_y': ctr_center_mass_y,
                             'ctr_center_mass_x': ctr_center_mass_x,
                             'corr_data_wz': corr_data_wz,
                             'corr_data_tw': corr_data_tw,
                             'corr_data_fwhm_hk': corr_data_fwhm_hk,
                             'corr_data_fwhm_l': corr_data_fwhm_l,
                             'time': time
                             })
# dump corrected data_frame
sample = files[0].split('_')[1]
data_df.to_pickle("attenuation_corrected_{0}.p".format(sample))

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
