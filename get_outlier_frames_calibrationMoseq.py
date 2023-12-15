# use moseq calibration results to get outlier frames from deeplabcut
# for refinement
import pickle
from utils import *
import os
import pandas as pd
from gaitAnalysis import DLCData
import pandas as pd
import glob

caliResults = r'Z:\HongliWang\openfield\katie\DLC6_6animals\outliersFrames.pickle'

with open(caliResults, 'rb') as f:
    data = pickle.load(f)

# rearrange the data into dataframe
outliers = {'session': [None]*len(data), 'frame':[None]*len(data)}

outlierDF = pd.DataFrame(outliers)
for idx in range(len(data)):
    outlierDF['session'].iloc[idx] = data[idx][0]
    outlierDF['frame'].iloc[idx] = data[idx][1]

videoPath = r'Z:\HongliWang\openfield\katie\DLC6_6animals\data'
sessions = np.unique(outlierDF['session'])
videoList = []
csvList = []
for ses in sessions:
    videoList.append(glob.glob(os.path.join(videoPath,'*'+ses[5:9]+'*.mp4'))[0])
    csvList.append(ses+'.csv')
sorted_session = sorted(sessions)
sorted_video = sorted(videoList)
sorted_csv = sorted(csvList)


# load dlc data
keypoints = {}
for idx in range(len(sorted_csv)):
    csvPath = os.path.join(videoPath,sorted_csv[idx])
    vPath = os.path.join(videoPath, sorted_video[idx])
    dlcResults = DLCData(csvPath, vPath, 40)
    keypoints[sorted_session[idx]] = dlcResults.data

# plot some frames with changepoints


for idx, ss in tqdm(enumerate(outlierDF['session'])):
    # read the bodyparts coordinates

    savefigpath = os.path.join(r'Z:\HongliWang\openfield\katie\DLC6_6animals\outlierplot', ss[5:9])
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)
    matplotlib.use('Agg')
    frame = read_video(glob.glob(os.path.join(videoPath,'*'+ss[5:9]+'*.mp4'))[0],
                outlierDF['frame'].iloc[idx], ifgray=False)
    #plt.figure()
    plt.imshow(frame)
    # plot keypoints
    for kp in keypoints[ss]['bodyparts']:
        plt.scatter(keypoints[ss][kp]['x'][outlierDF['frame'].iloc[idx]],
                keypoints[ss][kp]['y'][outlierDF['frame'].iloc[idx]], s = 50)
    figName = 'Frame' + str(outlierDF['frame'].iloc[idx]) + '.png'
    plt.savefig(os.path.join(savefigpath,figName))
    plt.close()

x = 1