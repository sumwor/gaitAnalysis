import os
import csv
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('QtAgg')  
import matplotlib.pyplot as plt

from tqdm import tqdm

plt.ion()

behCSV = r'Z:\HongliWang\Juvi_ASD Deterministic\BehaviorVideo\testVideo110525\ASD530_AB_rwdsz3_20251105.csv'
behVideoTimeStamp = r'Z:\HongliWang\Juvi_ASD Deterministic\BehaviorVideo\testVideo110525\frame_test_110525_Box8_1.csv'
ttlTimeStamp = r'Z:\HongliWang\Juvi_ASD Deterministic\BehaviorVideo\testVideo110525\ttl_tme_test_110525_Box_8_1.csv'
ttlMat = r'Z:\HongliWang\Juvi_ASD Deterministic\BehaviorVideo\testVideo110525\matrix_test_110525_Box_8_12025-11-05T10_50_57.8614272-08_00'

behMat = pd.read_csv(behCSV)
behVideoTS = pd.read_csv(behVideoTimeStamp, header=None)
ttlTS = pd.read_csv(ttlTimeStamp, header=None)

# read binary data (replace dtype/shape with yours)
data = np.fromfile(ttlMat, dtype=np.float32)

# reshape if you know the number of channels/columns

data = data.reshape(-1, 2)

# count times when TTL signal is larger than 2
valveMask = data[:,1]>2

# interpolate the time stamps for each sample
ttlTS_np = ttlTS.to_numpy().flatten() 
fs = 1000         # Hz
block_size = 1000 # samples per block

# Per-sample time offsets (0 to just before 1 second)
within_block = np.arange(block_size) / fs  # shape: (1000,)

# Broadcast addition to make 22531×1000 array
interp_ttlTS = ttlTS_np[:, None] + within_block[None, :]
interp_ttlTS = interp_ttlTS.ravel()

above_thresh = data[:,1] > 2
valve_on_times = np.where((~above_thresh[:-1]) & (above_thresh[1:]))[0] + 1
valve_on_times = valve_on_times[::3]
left_valve_timeStamp = interp_ttlTS[valve_on_times]

#%% find the valve on time from behavior csv

leftRewardMask = np.logical_and(behMat['schedule']==1, ~np.isnan(behMat['reward']))
left_valve_timeStamp_beh = behMat['outcome'][leftRewardMask].to_numpy()


#%% check the frame rate of the video
video_ts = np.array(behVideoTS).flatten()  
video_ts = video_ts/1000
frame_intervals = np.diff(video_ts)
rt_frame_rate = 1 / frame_intervals



# Define bin edges (1-second bins)
time_bins = np.arange(video_ts[0], video_ts[-1] + 1, 1)

# Count number of frames in each bin
frame_counts, _ = np.histogram(video_ts, bins=time_bins)

# Optional: average frame rate per bin (frames per second)
avg_frame_rate = frame_counts.astype(float)  # already frames per second



plt.show(block=False)