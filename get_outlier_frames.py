# use moseq changepoint results to get outlier frames from deeplabcut
# for refinement
import pickle
from utils import *
import os
import pandas as pd
from gaitAnalysis import DLCData

changepointResults = r'Z:\HongliWang\openfield\katie\DLC_2\changepoint\changepoints.pickle'

with open(changepointResults, 'rb') as f:
    data = pickle.load(f)

sessions = list(data['changepoints'].keys())
videoList = []
csvList = []
for ses in sessions:
    videoList.append(ses+'_labeled.mp4')
    csvList.append(ses+'.csv')
sorted_video = sorted(videoList)
sorted_csv = sorted(csvList)
videoPath = r'Z:\HongliWang\openfield\katie\DLC3\data'


# plot some frames with changepoints
sorted_ses = sorted(sessions)
for idx, ss in enumerate(sorted_ses):
    print("Working on session " + ss)

    numOutliers = len(data['changepoints'][ss])

    # load dlc points
    csvPath = os.path.join(videoPath,sorted_csv[idx])
    vPath = os.path.join(videoPath, sorted_video[idx])
    dlcResults = DLCData(csvPath, vPath, 40)
    keypoints = dlcResults.data
    # read the bodyparts coordinates
    print('Number of outlier frames ' + str(numOutliers))
    savefigpath = os.path.join(r'Z:\HongliWang\openfield\katie\DLC_2\outlierplot', ss[5:9])
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)
    matplotlib.use('Agg')
    for ii in tqdm(range(numOutliers)):
        if ii%5 == 0:
            frame = read_video(os.path.join(videoPath, sorted_video[idx]),
                       data['changepoints'][ss][ii], ifgray=False)
    #plt.figure()
            plt.imshow(frame)
         # plot keypoints
            for kp in keypoints['bodyparts']:
                plt.scatter(keypoints[kp]['x'][data['changepoints'][ss][ii]],
                        keypoints[kp]['y'][data['changepoints'][ss][ii]], s = 50)
            figName = 'Frame' + str(data['changepoints'][ss][ii]) + '.png'
            plt.savefig(os.path.join(savefigpath,figName))
            plt.close()

x = 1