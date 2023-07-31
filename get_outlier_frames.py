# use moseq changepoint results to get outlier frames from deeplabcut
# for refinement
import pickle
from utils import *
import os

changepointResults = r'D:\MoseqModel\openfield_cntnap_new\changepoint.pickle'

with open(changepointResults, 'rb') as f:
    data = pickle.load(f)

sessions = list(data.keys())
videoList = []
for ses in sessions:
    videoList.append(ses+'_labeled.mp4')

videoPath = r'Z:\HongliWang\DLCModels\openfield-Hongli-2023-06-09\videos'


# plot some frames with changepoints
for ss in sessions:
    numOutliers = len(data[ss]['changepoints'])
    savefigpath = os.path.join(r'D:\videos\outlierplot', ss[0:5])
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)
    matplotlib.use('Agg')
    for ii in tqdm(range(numOutliers)):
        frame = read_video(os.path.join(videoPath, videoList[0]),
                       data[ss]['changepoints'][ii], ifgray=False)
    #plt.figure()
        plt.imshow(frame)
        figName = 'Frame' + str(data[sessions[0]]['changepoints'][ii]) + '.png'
        plt.savefig(os.path.join(savefigpath,figName))
        plt.close()

x = 1