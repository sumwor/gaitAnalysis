# read pose-estimation result from deeplabcut csv file
# parameters to calculate:
# reference: sheppard, Kumar, 2022, Cell Reports
# 1. Angular velocity: vector base of tail to base of neck, degree/s
# 2. speed: movement speed of base of tail
# 3. acceleration
# 4. distance traveled: distance of base of tail
# 5. head direction: triangle of nose, left ear, right ear
# 6. body direction: spine 1 (base of neck) to tail (base of tail)
# 7. time in center zone and outer zone (the area of zones need to be determined session by session
#    (consider adding a reference to make sure the camera is always in place)
# to be added in Deeplabcut:
# 1. stride length (two hind paws at least)
# 2. step length
# 3. step width
# 4. nose/tail displacement
# 5.


""" questions for openfield behavior
1. individual variance v.s. genotype variance"""
import csv
import os.path

import pandas as pd
import glob
import numpy as np
from matplotlib import pyplot as plt
import imageio
from natsort import natsorted
import fitz
#from PIL import Image

from tqdm import tqdm
from pyPlotHW import *
from utils import *
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from tqdm import tqdm
from utility_HW import bootstrap
import h5py

import io
import subprocess as sp
import multiprocessing
import concurrent.futures
import functools

class DLCData:

    def __init__(self, filePath, videoPath, fps):
        self.filePath = filePath
        self.videoPath = videoPath
        self.nFrames = 0
        self.fps = fps
        # read data
        self.data = self.read_data()
        #self.video = self.read_video()

    def get_confidence(self, p_threshold, savefigpath):
        # get potentially outlier frames by confidence
        # focus on body parts that are most likely to be wrong:
        # tail 1, nose, left/right hand/foot
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'spine 1', 'spine 2', 'spine 3', 'tail 1']
        outliers = []
        for f in range(self.nFrames):
            for bp in body_parts:
                if self.data[bp]['p'][f]<p_threshold:
                    outliers.append(f)
                break

        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
        #self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath,figName))
            plt.close()

    def get_jump(self, px_threshold, savefigpath):
        body_parts = ['nose', 'left hand', 'right hand', 'left foot', 'right foot', 'tail 1', 'tail 2', 'tail 3']
        outliers = []
        for f in range(self.nFrames-1):
            for bp in body_parts:
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                if np.sqrt(dx2+dy2) > px_threshold:
                    outliers.append(f+1)
                break


        for ff in tqdm(range(len(outliers))):
            frame = read_video(videoPath, outliers[ff], ifgray=False)
            # self.plot_frame_label(outliers[1])
            plt.imshow(frame)
            figName = 'Frame' + str(outliers[ff]) + '.png'
            plt.savefig(os.path.join(savefigpath, figName))
            plt.close()

    def kp_jump_dist(self):
        # calculate cross frame keypoint jumps and plot the distrubution
        body_parts = self.data['bodyparts']
        kp_jumps = {}
        for f in range(self.nFrames-1):
            for bp in body_parts:
                if f==0:
                    kp_jumps[bp] = []
                dx2 = (self.data[bp]['x'][f+1]-self.data[bp]['x'][f])**2
                dy2 = (self.data[bp]['y'][f+1]-self.data[bp]['y'][f])**2
                kp_jumps[bp].append(np.sqrt(dx2+dy2))

        bins = np.arange(0.0,30,0.2)
        fig, ax = plt.subplots(3,5,sharey=True)
        for idx, bp in enumerate(body_parts):
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].hist(kp_jumps[bp], bins= bins)
            ax[int(np.floor(idx/5)), int(np.mod(idx,5))].set_xlabel(bp)

    def moving_trace(self, savefigpath):
        """ plot animal moving trace in the field"""
        if not hasattr(self, 'arena'):
            self.arena = frame_input(self.videoPath)

        arena_x = [self.arena['upper left'][0], self.arena['upper right'][0],
                   self.arena['lower right'][0], self.arena['lower left'][0], self.arena['upper left'][0]]
        arena_y = [self.arena['upper left'][1], self.arena['upper right'][1],
                   self.arena['lower right'][1], self.arena['lower left'][1], self.arena['upper left'][1]]
        tracePlot = StartPlots()
        tracePlot.ax.plot(arena_x, arena_y)
        tracePlot.ax.plot(self.data['tail 1']['x'], self.data['tail 1']['y'])
        tracePlot.ax.axis('equal')
        # Hide the x and y axes
        tracePlot.ax.axis('off')
        tracePlot.save_plot('Moving trace.tif', 'tif', savefigpath)
        tracePlot.save_plot('Moving trace.svg', 'svg', savefigpath)

    def get_time_in_center(self):
        """calculate time spent in the center"""
        if hasattr(self, 'arena'):
            # do the calculation
            side_length = np.sqrt((self.arena['upper left'][0] - self.arena['upper right'][0])**2 + (self.arena['upper left'][1] - self.arena['upper right'][1])**2)
            self.center = {}
            self.center['upper left'] = (self.arena['upper left'][0] + side_length/4,
                                         self.arena['upper left'][1] + side_length/4)
            self.center['upper right'] = (self.arena['upper right'][0] - side_length/4,
                                          self.arena['upper right'][1] + side_length / 4)
            self.center['lower right'] = (self.arena['lower right'][0] - side_length/4,
                                             self.arena['lower right'][1] - side_length / 4)
            self.center['lower left'] = (self.arena['lower left'][0] + side_length/4,
                                            self.arena['lower right'][1] - side_length / 4)

            # determine if tail 1 is inthe center area\
            x_left = (self.center['upper left'][0] + self.center['lower left'][0])/2
            x_right = (self.center['upper right'][0] + self.center['lower right'][0])/2
            y_upper = (self.center['upper left'][1] + self.center['upper right'][1])/2
            y_lower = (self.center['lower left'][1] + self.center['lower right'][1])/2

            is_center = np.zeros(len(self.data['tail 1']['x']))
            for idx in range(len(self.data['tail 1']['x'])):
                if self.data['tail 1']['x'][idx] > x_left and self.data['tail 1']['x'][idx] < x_right:
                    if self.data['tail 1']['y'][idx] > y_upper and self.data['tail 1']['y'][idx] < y_lower:
                        is_center[idx] = 1

            self.time_in_center = is_center
            self.cumu_time_center = []
            cumu = 0
            for f in range(self.nFrames):
                cumu += self.time_in_center[f]/self.fps
                self.cumu_time_center.append(cumu)

        else:
            print("please run moving_trace first")

    #def plot_frame_label(self, frame):
    #    image = read_video(self.videoPath, frame, ifgray = False)
    #    labelPlot = StartPlots()
    #    labelPlot.ax.imshow(image)
    #    for bb in self.data['bodyparts']:
    #        labelPlot.ax.scatter(self.data[bb]['x'][frame], self.data[bb]['y'][frame])

    #    labelPlot.legend(self.data['bodyparts'])

    def read_data(self):
        data = {}
        self.t = []
        with open(self.filePath) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # scorer
                    data[row[0]] = row[1]
                    line_count += 1
                elif line_count == 1:  # body parts
                    bodyPartList = []
                    for bb in range(len(row) - 1):
                        if row[bb + 1] not in bodyPartList:
                            bodyPartList.append(row[bb + 1])
                    data[row[0]] = bodyPartList
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 2:  # coords
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                elif line_count == 3:  # actual coords
                    # print({", ".join(row)})
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        if np.mod(ii, 3) == 0:
                            data[body] = {}
                        data[body][tempList[np.mod(ii, 3)]] = [float(row[ii + 1])]
                    self.t.append(0)
                    line_count += 1
                    self.nFrames += 1

                else:
                    tempList = ['x', 'y', 'p']
                    for ii in range(len(row) - 1):
                        # get the corresponding body parts based on index
                        body = data['bodyparts'][int(np.floor((ii) / 3))]
                        data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                    self.t.append(self.nFrames*(1/self.fps))
                    line_count += 1
                    self.nFrames += 1

            print(f'Processed {line_count} lines.')

            # add frame time
            tStep= 1/self.fps
            data['time'] = np.arange(0, tStep*self.nFrames, tStep)
            self.t = np.array(self.t)
        return data

    def check_quality(self):
        # go through the data and plot the distribution of p-values
        pass

    def get_movement(self):
        # calculate distance, running velocity, acceleration, based on tail (base of tail)
        self.vel = np.zeros((self.nFrames-1, 1))
        self.dist = np.zeros((self.nFrames-1,1))
        self.accel = np.zeros((self.nFrames-1, 1))

        for ff in range(self.nFrames-1):
            self.dist[ff] = np.sqrt((self.data['tail 1']['x'][ff+1] - self.data['tail 1']['x'][ff])**2 +
                (self.data['tail 1']['y'][ff + 1] - self.data['tail 1']['y'][ff]) ** 2)

            self.vel[ff] = (self.dist[ff])*self.fps
            if ff<self.nFrames-2:
                self.accel[ff] = (self.vel[ff+1]-self.vel[ff])*self.fps

    def get_angular_velocity(self):
        # calculate angular velocity based on tail and spine 1
        self.angVel = np.zeros((self.nFrames-1, 1))
        for ff in range(self.nFrames-1):
            y1 = self.data['spine 1']['y'][ff] - self.data['tail 1']['y'][ff]
            x1 = self.data['spine 1']['x'][ff] - self.data['tail 1']['x'][ff]

            y2 = self.data['spine 1']['y'][ff+1] - self.data['tail 1']['y'][ff+1]
            x2 = self.data['spine 1']['x'][ff+1] - self.data['tail 1']['x'][ff+1]

            self.angVel[ff] = self.get_angle([x1, y1], [x2, y2])*self.fps
        #self.angVel = self.angVel*self.fps

    def get_head_angular_velocity(self):
        self.headAngVel = np.zeros((self.nFrames, 1))
        for ff in range(self.nFrames-1):
            # get the mid point of two ears
            midX1 = (self.data['left ear']['x'][ff] + self.data['right ear']['x'][ff])/2
            midY1 = (self.data['left ear']['y'][ff] + self.data['right ear']['y'][ff])/2

            midX2 = (self.data['left ear']['x'][ff+1] + self.data['right ear']['x'][ff+1])/2
            midY2 = (self.data['left ear']['y'][ff+1] + self.data['right ear']['y'][ff+1])/2

            v1 = [self.data['nose']['x'][ff]-midX1, self.data['nose']['y'][ff]-midY1]
            v2 = [self.data['nose']['x'][ff+1]-midX2, self.data['nose']['y'][ff+1]-midY2]

            self.headAngVel[ff] = self.get_angle(v1, v2) * self.fps

    def get_angle(self, v1, v2):
        # get angle between two vectors
            v1_u = self.unit_vector(v1)
            v2_u = self.unit_vector(v2)

            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            if v1[0] * v2[1] - v1[1] * v2[0] < 0:
                angle = -angle
            return angle

    def unit_vector(self, v):
        """ Returns the unit vector of the vector.  """
        return v / np.linalg.norm(v)

    # def generate_videos(self, timeInterval, videoFilePath):
    #     # generate pose estimation videos with velocity, head direction, angular velocity ploted simutaneously
    #     # timeInterval: time interval for the video
    #     # get number of frames based on timeInterval
    #     frameIdx = np.arange(self.nFrames)
    #     frames = frameIdx[np.logical_and(self.t>=timeInterval[0],
    #         self.t<timeInterval[-1])]
    #     plotTime = self.t[np.logical_and(self.t>=timeInterval[0],
    #         self.t<timeInterval[-1])]
    #     frameCount = 0
    #     videoFrames = []
    #
    #     labelPlot = StartSubplots(4, 1, gridspec_kw=[1, 1, 1, 20])
    #
    #     writer = animation.FFMpegWriter(fps = 40)
    #     with writer.saving(labelPlot.fig, videoFilePath, dpi = 100):
    #         for f in tqdm(frames):
    #             # clear the last frame
    #             for ax in labelPlot.ax:
    #                 ax.clear()
    #         #gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,20])
    #             image = read_video(f, ifgray = True)
    #                    #labelPlot.ax[3].subplot(gs[3])
    #             labelPlot.ax[3].imshow(image, cmap = 'gray')
    #             for bb in self.data['bodyparts']:
    #                 labelPlot.ax[3].scatter(self.data[bb]['x'][f],
    #                                  self.data[bb]['y'][f])
    #             #labelPlot.legend(3, 0, self.data['bodyparts'])
    #
    #             # plot velocity
    #             labelPlot.ax[0].plot(plotTime[0:frameCount+1],
    #                              self.vel[frames[0:frameCount+1]], label='velocity')
    #             labelPlot.ax[0].set_ylabel('Vel')
    #             labelPlot.ax[0].set_xlim(plotTime[0], plotTime[-1])
    #             labelPlot.ax[0].set_ylim(min(self.vel[frames]),
    #                                      max(self.vel[frames]))
    #             labelPlot.ax[1].plot(plotTime[0:frameCount+1],
    #                              self.angVel[frames[0:frameCount+1]], label='angular velocity')
    #             labelPlot.ax[1].set_ylabel('BD')
    #             labelPlot.ax[1].set_xlim(plotTime[0], plotTime[-1])
    #             labelPlot.ax[1].set_ylim(-np.pi, np.pi)
    #             labelPlot.ax[2].plot(plotTime[0:frameCount+1],
    #                              self.headAngVel[frames[0:frameCount+1]], label='head direction')
    #             labelPlot.ax[2].set_ylabel('HD')
    #             labelPlot.ax[2].set_xlim(plotTime[0], plotTime[-1])
    #             labelPlot.ax[2].set_ylim(-np.pi, np.pi)
    #
    #             frameCount += 1
    #             writer.grab_frame()

class Moseq:

    def __init__(self, root_dir):
        # read

        self.filePath = os.path.join(root_dir,'Data','Moseq', 'results.h5')
        self.data = self.read_data()

    def read_data(self):
        data = {}

        with h5py.File(self.filePath, 'r+') as f:
            # behavior sessions included in the results
            self.sessions = list(f.keys())

            # fitted parameters in the results
            params = list(f[self.sessions[0]])
            self.params = params

            for ses in self.sessions:
                data[ses] = {}
                for par in params:
                    data[ses][par] = np.array(f[ses][par])

        return data

    def get_syllables(self, DLCSum):
        # clean the data, get syllable transition vector

        for idx, ses in enumerate(self.sessions):
            dlcObj = DLCSum.data['DLC_obj'][idx]
            syl = self.data[ses]['syllables_reindexed']
            self.data[ses]['next_syllable'] = [np.nan for nn in range(len(self.data[ses]['syllables_reindexed']))]
            self.data[ses]['prev_syllable'] = [np.nan for nn in range(len(self.data[ses]['syllables_reindexed']))]

            if idx == 0:
                self.all_syllables = np.unique(syl)
            else:
                self.all_syllables = np.unique(np.concatenate((syl,self.all_syllables)))
            syllable_tran = []
            syllable_dur = []
            transition_time = []
            prev_s = np.nan
            prev_syllable = prev_s
            syl_count = 1
            for idx, s in enumerate(syl):
                self.data[ses]['prev_syllable'][idx] = prev_syllable
                if np.isnan(prev_s):
                    syllable_tran.append(s)
                    transition_time.append(0)
                else:
                    if s == prev_s:
                        syl_count += 1
                    else:
                        prev_syllable = prev_s
                        self.data[ses]['next_syllable'][idx-syl_count:idx]=[s]*syl_count
                        syllable_dur.append(syl_count)
                        syl_count = 1
                        syllable_tran.append(s)
                        transition_time.append(dlcObj.t[idx])
                prev_s = s
            syllable_dur.append(syl_count)  # add the duration of the last syllable
            syllable_dur = np.array(syllable_dur) / fps  # convert to seconds
            self.data[ses]['syllable_transition'] = syllable_tran
            self.data[ses]['syllable_duration'] = syllable_dur
            self.data[ses]['transition_time'] = transition_time

        # calculate overall syllable appearance
        syl_appear_all = []
        for syl in self.all_syllables:
            syl_count = 0
            for ses in self.sessions:
                syl_count += np.sum(self.data[ses]['syllable_transition'] == syl)
            syl_appear_all.append(syl_count)

        self.syl_appear_all = syl_appear_all

    def load_syllable_plot(self, root_dir):
        # change to moseq_folder first
        syllable_plots_dir = os.path.join(root_dir, 'Data', 'Moseq', 'trajectory_plots')
        filePattern = "Syllable*.gif"
        syllable_plot=glob.glob(f"{syllable_plots_dir}/{filePattern}")
        self.syllable_plots_path = natsorted(syllable_plot)
        self.syllable_plots = []
        for path in self.syllable_plots_path:
            syl_images = imageio.mimread(path)
            self.syllable_plots.append(syl_images[-1])

    def process_video_chunk(self, chunkIdx, params):
        matplotlib.use('Agg')
        speed = params['speed']
        session = params['session']
        dlcObj = params['dlcObj']
        videopath = params['videopath']
        frameRange = params['frameRange']

        processed_chunk = []

        frames = [c for c in chunkIdx]
        syllables = self.data[session]['syllables_reindexed']
        frameCount = 0
        syllable_count = 0

        colormap_name = 'viridis'  # Replace with the desired colormap name
        # Get the colormap
        cmap = cm.get_cmap(colormap_name)

        num_syllables = len(self.all_syllables)
        colors_syllable = [cmap(x) for x in np.linspace(0, 1, num_syllables)]

        temp_trans = np.array(self.data[session]['transition_time'])

        bodyparts = dlcObj.data['bodyparts']
        skeleton = [
            ['nose', 'head'],
            ['head', 'left ear'],
            ['head', 'right ear'],
            ['head', 'spine 1'],
            ['spine 1', 'left hand'],
            ['spine 1', 'right hand'],
            ['spine 1', 'spine 2'],
            ['spine 2', 'spine 3'],
            ['spine 3', 'left foot'],
            ['spine 3', 'right foot'],
            ['spine 3', 'tail 1'],
            ['tail 1', 'tail 2'],
            ['tail 2', 'tail 3']
        ]

        colors = [cmap(x) for x in np.linspace(0, 1, len(bodyparts))]

        fig = plt.figure(tight_layout=True, figsize=(10, 10))
        gs = gridspec.GridSpec(7, 3)

        for f in tqdm(chunkIdx):
            plt.clf()
            # clear the last frame
            ax=fig.add_subplot(gs[0,:])
            ax.set_aspect(1/15)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
             # Hide the ticks and tick labels
            ax.set_yticks([])
            # plot the syllable bars
            plotWindow = 160 #plot syllables in 400 frames
            startFrame = int(f-plotWindow/2)
            endFrame = int(f+plotWindow/2)
            syllable_window = np.zeros(plotWindow+1)
            syllable_window[:]=np.nan
            syllable_time = np.arange(startFrame/dlcObj.fps,(endFrame+1)/dlcObj.fps,1/dlcObj.fps)

            if startFrame < 0 and endFrame <= dlcObj.nFrames:
                syllable_window[-startFrame+1:] = syllables[0:endFrame]
                    #syllable_time[-startFrame+1:] = plotTime[0:endFrame]
            elif startFrame >= 0 and endFrame > dlcObj.nFrames:
                syllable_window[:plotWindow-endFrame+len(frames)] = syllables[startFrame:]
                    #syllable_time[:plotWindow-endFrame+len(frames)] = plotTime[startFrame:]
            elif startFrame < 0 and endFrame > dlcObj.nFrames:
                syllable_window[-startFrame:plotWindow - endFrame + len(frames)] = syllables[:]
                    #syllable_time[-startFrame:plotWindow - endFrame + len(frames)] = plotTime[startFrame:]
            else:
                syllable_window = syllables[startFrame:endFrame]
                #syllable_time = plotTime[startFrame:endFrame]

            ax.plot([f/dlcObj.fps,f/dlcObj.fps],[-0.5,2.5])
            # plot transition point
            trans_included = temp_trans[np.logical_and(temp_trans >= startFrame/dlcObj.fps,
                                                           temp_trans< endFrame/dlcObj.fps)]
            for t in trans_included:
                idx = self.data[session]['transition_time'].index(t)
                ax.plot([t+0.5/dlcObj.fps, t+0.5/dlcObj.fps], [-0.5, 2.5], color='black', linewidth=1)
                #ax.text(0,0,'test', fontfamily='serif')
                ax.text(t-0.5/dlcObj.fps, 3.5,
                            '#'+str(self.data[session]['syllable_transition'][idx]),
                            fontdict={'weight': 'bold', 'size': 8},  fontfamily='serif')
                ax.text(t-0.5/dlcObj.fps, 2.5,
                            str(int(self.data[session]['syllable_duration'][idx]*1000)),
                            fontdict={'weight': 'bold', 'size': 8},  fontfamily='serif')
            for idx, sl in enumerate(syllable_window):
                if not np.isnan(sl):
                    ax.bar(syllable_time[idx], 2, 1/dlcObj.fps,
                               color=colors_syllable[list(self.all_syllables).index(sl)])
            ax.set_xlim(syllable_time[0], syllable_time[-1])

            """subplot 2"""
            image = read_video(videopath, f, ifgray = True)
                        #labelPlot.ax[3].subplot(gs[3])
            ax_frame = fig.add_subplot(gs[1:-1,0:2])
            ax_frame.set_aspect('equal')
                #ax_frame.clear()
            ax_frame.imshow(image, cmap = 'gray')

            ax_frame.spines['top'].set_visible(False)
            ax_frame.spines['right'].set_visible(False)
            ax_frame.spines['bottom'].set_visible(False)
            ax_frame.spines['left'].set_visible(False)

            # Hide the ticks and tick labels
            ax_frame.set_xticks([])
            ax_frame.set_yticks([])

            for idx, bp in enumerate(bodyparts):
                ax_frame.scatter(dlcObj.data[bp]['x'][f],
                                dlcObj.data[bp]['y'][f], s=40, color=colors[idx])
                for skel in skeleton:
                    if bp in skel:
                        ax_frame.plot([dlcObj.data[skel[0]]['x'][f],
                                      dlcObj.data[skel[1]]['x'][f]],
                                     [dlcObj.data[skel[0]]['y'][f],
                                      dlcObj.data[skel[1]]['y'][f]], color='black')

            """subplot 3,4,5
            prev_syllable, curr_syllable, next_syllable"""

            prev_syllable = self.data[session]['prev_syllable'][f]
            curr_syllable = self.data[session]['syllables_reindexed'][f]
            next_syllable = self.data[session]['next_syllable'][f]

            ax_prev = fig.add_subplot(gs[1:3,2])
            ax_prev.set_aspect('equal')
            ax_prev.set_title('Previous syllable')
            ax_prev.axis('off')
            if not np.isnan(prev_syllable):
                prev_idx = list(self.all_syllables).index(prev_syllable)
                ax_prev.imshow(self.syllable_plots[prev_idx])

            ax_curr = fig.add_subplot(gs[3:5,2])
            ax_curr.set_aspect('equal')
            curr_idx = list(self.all_syllables).index(curr_syllable)
            ax_curr.imshow(self.syllable_plots[curr_idx])
            ax_curr.axis('off')
            ax_curr.set_title('Current syllable')

            ax_next = fig.add_subplot(gs[5:7,2])
            ax_next.set_aspect('equal')
            ax_next.axis('off')
            ax_next.set_title('Next syllable')
            if not np.isnan(next_syllable):
                next_idx = list(self.all_syllables).index(next_syllable)
                ax_next.imshow(self.syllable_plots[next_idx])

            # save the figure then read it
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_frame = imageio.imread(buf)

            processed_chunk.append(plot_frame)

        return processed_chunk

    def multiprocess_videos(self,speed, session, frameRange, dlcObj, videopath, savevideopath):

        total_frames = len(frameRange)
        num_processors = multiprocessing.cpu_count()
        chunk_size = total_frames//num_processors
        if np.mod(total_frames, num_processors) == 0:
            chunks = [list(np.arange(i, i + chunk_size)) for i in range(frameRange[0], frameRange[0]+total_frames, chunk_size)]
        else:
            chunks = [list(np.arange(i,i+chunk_size)) for i in range(frameRange[0], frameRange[0]+total_frames-chunk_size, chunk_size)]
        if chunks[-1][-1]+1 < frameRange[0]+total_frames:
            chunks[-1]+=(list(np.arange(chunks[-1][-1]+1, total_frames)))

        # parallel computing
        params = {}
        params['speed'] = speed
        params['session'] = session
        params['dlcObj'] = dlcObj
        params['videopath'] = videopath
        params['frameRange'] = frameRange

        partial_process_chunk = functools.partial(self.process_video_chunk, params=params)

        with multiprocessing.Pool(processes=num_processors) as pool:
            processed_chunks = pool.map(partial_process_chunk, chunks)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processors) as executor:
        #     process_func = functools.partial(self.process_video_chunk, params=params)
        #     processed_chunks = list(executor.map(process_func, chunks))

        # processed_frames = np.concatenate(processed_chunks)

        writer = imageio.get_writer(savevideopath, fps=dlcObj.fps*speed)
        for processed_frames in processed_chunks:
            for frame in tqdm(processed_frames):
                writer.append_data(frame)
        writer.close()

    def get_velocity(self, session, fps):

        nFrames = len(self.data[session]['centroid'])
        vel = np.zeros((nFrames-1, 1))
        dist = np.zeros((nFrames-1, 1))
        for ff in range(nFrames-1):
            dist[ff] = np.sqrt((self.data[session]['centroid'][ff+1,0] - self.data[session]['centroid'][ff,0])**2 +
                (self.data[session]['centroid'][ff+1, 1] - self.data[session]['centroid'][ff,1]) ** 2)
            vel[ff] = dist[ff]*fps

        return vel

    def syllable_frequency(self,DLCSum, savefigpath):
        # analyze the syllables
        # 1.overall frequency of every syllable
        # 2. for each syllables:
        #    2.1  appearance overtime
        #    2.2  duration distribution
        for sesIdx,ses in enumerate(self.sessions):
            syllables = self.data[ses]['syllables_reindexed']
            syl_frequency = np.zeros((len(self.all_syllables),1))
            syl_freq_app = np.zeros((len(self.all_syllables), 1))
            for idx, syl in enumerate(self.all_syllables):
                syl_frequency[idx] = np.sum(syllables==syl)/len(syllables)
                syl_freq_app[idx] = np.sum((self.data[ses]['syllable_transition'] == syl)) / len(
                    self.data[ses]['syllable_transition'])

            # plot and save
            savefig = os.path.join(savefigpath,DLCSum.animals[sesIdx])
            if not os.path.exists(savefig):
                os.makedirs(savefig)

            """plot overall frequency of each syllable"""
            freqAllPlot = StartPlots()
            freqAllPlot.ax.plot(self.all_syllables, syl_frequency)
            freqAllPlot.ax.set_xlabel('Syllable #')
            freqAllPlot.ax.set_ylabel('Frequency')
            freqAllPlot.ax.plot(self.all_syllables, syl_freq_app)
            freqAllPlot.ax.legend(['By frames','By appearances' ])
            freqAllPlot.ax.set_title(DLCSum.animals[sesIdx])
            freqAllPlot.save_plot('Overall frequency of syllables.tif','tif', savefig)

            # duration distribution of each syllable
            maxDur = 10
            timeStep = 0.05
            timeBin = np.arange(0, maxDur, timeStep)
            syllable_duration = np.zeros((len(self.all_syllables), len(timeBin)))
            syl_idx = np.arange(len(self.data[ses]['syllable_transition']))
            totalAppearance = len(self.data[ses]['syllable_transition'])
            for idx, syl in enumerate(self.all_syllables):
                syl_appear = syl_idx[self.data[ses]['syllable_transition'] == syl]
                for app in syl_appear:
                    dur = self.data[ses]['syllable_duration'][app]
                    timeIndex = int(np.floor(dur / timeStep))
                    syllable_duration[idx, timeIndex] += 1
            # make plot
            matplotlib.use('Agg')
            figpath_dist = os.path.join(savefig,'Individual_dur_distribution')
            if not os.path.exists(figpath_dist):
                os.makedirs(figpath_dist)
            for ii in tqdm(range(len(self.all_syllables))):
                fig_name = 'Duration distribution of syllable '+str(self.all_syllables[ii])
                distFig = StartPlots()
                distFig.ax.bar(timeBin[0:40], syllable_duration[ii, 0:40] / totalAppearance, width=0.025)
                distFig.ax.set_xlabel('Time (s)')
                distFig.ax.set_ylabel('Frequency')
                distFig.ax.set_title('Syllable '+str(self.all_syllables[ii]))
                distFig.save_plot(fig_name+'.tif','tif',figpath_dist)

            # appearance across time
            figpath_appear = os.path.join(savefig,'Individual_appearance')
            if not os.path.exists(figpath_appear):
                os.makedirs(figpath_appear)

            maxDur = DLCSum.plotT[-1]
            timeStep = 15
            timeBin = np.arange(0, maxDur, timeStep)
            syllable_appear = np.zeros((len(self.all_syllables), len(timeBin)))
            totalAppearance = len(self.data[ses]['syllable_transition'])
            for idx, syl in enumerate(self.all_syllables):
                syl_appear = syl_idx[self.data[ses]['syllable_transition'] == syl]
                for app in syl_appear:
                    startTime = self.data[ses]['transition_time'][app]
                    timeIndex = int(np.floor(startTime / timeStep))
                    syllable_appear[idx, timeIndex] += 1

            for ii in tqdm(range(len(self.all_syllables))):
                fig_name = 'Frequency across session '+str(self.all_syllables[ii])
                freqFig = StartPlots()
                freqFig.ax.plot(timeBin, syllable_appear[ii,:] / totalAppearance)
                freqFig.ax.set_xlabel('Time (s)')
                freqFig.ax.set_ylabel('Frequency')
                freqFig.ax.set_title('Syllable '+str(self.all_syllables[ii]))
                freqFig.save_plot(fig_name+'.tif','tif',figpath_appear)

    def syllable_transition(self, DLCSum, savefigpath):
        # plot syllable transition matrix
        # Calculate the transition counts
        for sesIdx,ses in enumerate(self.sessions):
            transition_counts = {}
            savefig = os.path.join(savefigpath,str(DLCSum.animals[sesIdx]))
            if not os.path.exists(savefig):
                os.mkdir(savefig)
            for i in range(len(self.data[ses]['syllable_transition']) - 1):
                current_state = self.data[ses]['syllable_transition'][i]
                next_state = self.data[ses]['syllable_transition'][i + 1]
                transition = (current_state, next_state)
                if transition in transition_counts:
                    transition_counts[transition] += 1
                else:
                    transition_counts[transition] = 1

            # Create the transition probability matrix
            unique_states = self.all_syllables
            num_states = len(unique_states)
            transition_matrix = np.zeros((num_states, num_states))
            for i, state1 in enumerate(unique_states):
                for j, state2 in enumerate(unique_states):
                    transition = (state1, state2)
                    if transition in transition_counts:
                        transition_matrix[i, j] = transition_counts[transition]

            # Normalize the transition matrix to obtain probabilities
            transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
            transition_matrix[np.isnan(transition_matrix)] = 0
            transPlot = StartPlots()
            im=transPlot.ax.imshow(transition_matrix,cmap = 'Reds')
            transPlot.fig.colorbar(im)
            transPlot.ax.set_xlabel('Next syllable')
            transPlot.ax.set_ylabel('Current syllable')
            transPlot.ax.set_title('Transition matrix '+ str(DLCSum.animals[sesIdx]))
            transPlot.save_plot('Transition matrix '+ str(DLCSum.animals[sesIdx])+'.tif',
                                                          'tif', savefig)
            # for Moseq analysis

    def get_ego_centric_coordinates(self):
        # get ego-centric coordinates
        for ss, session in enumerate(self.session):
            self.data[session]['ego_centric'] = np.zeros((self.data[session]['estimated_coordinates'].shape[0],
                                                          len(DLCSum.data['DLC_obj'][ss].data['bodyparts']),2))
            coordinates = DLCSum.data['DLC_obj'][ss].data
            bodyparts = DLCSum.data['DLC_obj'][ss].data['bodyparts']
            for idx in range(self.data[session]['ego_centric'].shape[0]):
                for jdx in range(self.data[session]['ego_centric'].shape[1]):
                    tempCoord = [coordinates[bodyparts[jdx]]['x'][idx],
                                 coordinates[bodyparts[jdx]]['y'][idx]]
                    translated_point = tempCoord-self.data[session]['centroid'][idx]
                    headDir = self.data[session]['heading'][idx]
                    rotation_mat = np.array([[np.cos(headDir), np.sin(headDir)],
                                            [-np.sin(headDir), np.cos(headDir)]])
                    rotated_point = np.dot(rotation_mat, translated_point)
                    self.data[session]['ego_centric'][idx, jdx,:] = rotated_point

            # plots
            frame = 1
            plt.scatter(self.data[session]['ego_centric'][frame, 0:3, 0],
                        self.data[session]['ego_centric'][frame, 0:3, 1])
            #skeleton = [[0,3],[1,3], [2,3], [6, 4], [6,5], [3,6], [6,7], [7,8], [8,9],
            #            [8,10], [8,11], [11,12], [12,13]]
            skeleton = [[0, 3], [1, 3], [2, 3]]
            for skel in skeleton:
                if skel[0]==0:
                    lineWidth = 2
                else:
                    lineWidth = 1
                plt.plot([self.data[session]['ego_centric'][frame , skel[0], 0],
                          self.data[session]['ego_centric'][frame , skel[1], 0]],
                         [self.data[session]['ego_centric'][frame , skel[0], 1],
                          self.data[session]['ego_centric'][frame , skel[1], 1]],
                         linewidth=lineWidth)


    def tail_dist(self, DLCSum ,savefigpath):
        # plot the density distribution of tail 3
        for ss,session in enumerate(self.sessions):
            for idx, bp in enumerate(DLCSum.data['DLC_obj'][ss].data['bodyparts']):
                dots = self.data[session]['ego_centric'][:,idx,:]
                x = self.data[session]['ego_centric'][:,idx,0]
                y = self.data[session]['ego_centric'][:, idx, 1]
            # Generate random data points for demonstration
            # Replace with your actual dot coordinates

            # Define the grid on which to evaluate the KDE
                grid_x = np.linspace(np.min(x), np.max(x), 100)
                grid_y = np.linspace(np.min(y), np.max(y), 100)
                grid_x, grid_y = np.meshgrid(grid_x, grid_y)
                grid_points = np.stack([grid_x, grid_y], axis=-1)

            # Calculate the KDE
                kde = np.zeros_like(grid_x)
                for dot in dots:
                    kde += np.exp(-0.5 * np.sum((grid_points - dot.T) ** 2, axis=2))

            # Normalize the KDE
                kde /= (2 * np.pi * np.std(x) * np.std(y))

            # Plot the density distribution
                plt.figure(figsize=(10, 8))
                plt.contourf(grid_x, grid_y, kde, levels=20, cmap='viridis')
                    plt.colorbar(label='Density')
            #plt.scatter(x, y, s=10, color='black', alpha=0.5)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Density Distribution')
                plt.show()
# generate a video clip with syllables marked on top?
#

class DLCSummary:
    """class to summarize DLC data and analysis"""

    def __init__(self, root_folder, animals, GeneBG, fps):
        self.rootFolder = root_folder
        self.dataFolder = os.path.join(root_folder, 'Data')
        self.analysisFolder = os.path.join(root_folder, 'Analysis')
        self.sumFolder = os.path.join(root_folder, 'Summary')
        self.animals = animals
        self.GeneBG = GeneBG
        self.fps = fps

        # make directories
        if not os.path.exists(self.analysisFolder):
            os.makedirs(self.analysisFolder)
        if not os.path.exists(self.sumFolder):
            os.makedirs(self.sumFolder)
        self.data = pd.DataFrame(animals, columns=['Animal'])
        DLC_results = []
        video = []
        for aa in animals:
            filePatternCSV = aa + '*.csv'
            filePatternVideo = aa + '*.mp4'
            DLC_results.append(glob.glob(f"{dataFolder}/{'DLC'}/{filePatternCSV}")[0])
            video.append(glob.glob(f"{dataFolder}/{'Videos'}/{filePatternVideo}")[0])
        self.data['CSV'] = DLC_results
        self.data['Video'] = video
        self.data['GeneBG'] = GeneBG

        self.nSubjects = len(animals)
        DLC_obj = []
        minFrames = 10 ** 8
        for s in range(self.nSubjects):
            filePath = self.data['CSV'][s]
            videoPath = self.data['Video'][s]
            dlc = DLCData(filePath, videoPath, fps)
            DLC_obj.append(dlc)
            if dlc.nFrames < minFrames:
                minFrames = dlc.nFrames

        self.minFrames = minFrames
        self.data['DLC_obj'] = DLC_obj
        self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSubjects)
        self.WTIdx = animalIdx[self.data['GeneBG'] == 'WT']
        self.MutIdx = animalIdx[self.data['GeneBG'] == 'Mut']

    def plot_outlier_frames(self):
        for idx, animal in enumerate(self.animals):
            dlc = self.data['DLC_obj'][idx]
            dlc.kp_jump_dist()
            savefigpath = os.path.join(r'D:\videos\outlierplot_confidence_3', animal)
            if not os.path.exists(savefigpath):
                os.mkdir(savefigpath)
            dlc.get_confidence(0.95, savefigpath)
            #savefigpath = os.path.join(r'D:\videos\outlierplot_jump_3', animal)
            #if not os.path.exists(savefigpath):
            #    os.mkdir(savefigpath)
            #dlc.get_jump(15, savefigpath)

    def motion_analysis(self, savefigpath):
        # basic analysis for motion related variables
        # distance traveled, speed, angular velocity...
        distanceMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        velocityMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)

        #
        velEdges = np.arange(0, 1000, 10)
        velocityDist = np.full((len(velEdges), self.nSubjects), np.nan)
        angEdges = np.arange(-15, 15, 0.5)
        angularDist = np.full((len(angEdges), self.nSubjects), np.nan)
        headAngularDist = np.full((len(angEdges), self.nSubjects), np.nan)

        for idx, obj in enumerate(self.data['DLC_obj']):
            obj.get_movement()
            # cumulative curve of distance travelled
            cumu_dist = np.cumsum(obj.dist)
            distanceMat[:, idx] = cumu_dist[0:self.minFrames - 1]
            velocityMat[:, idx] = obj.vel[0:self.minFrames - 1, 0]
            counts, _ = np.histogram(obj.vel, bins=velEdges)
            velocityDist[0:-1, idx] = counts * 100 / (sum(counts))

            obj.get_angular_velocity()
            counts, _ = np.histogram(obj.angVel, bins=angEdges)
            angularDist[0:-1, idx] = counts * 100 / (sum(counts))

            obj.get_head_angular_velocity()
            counts, _ = np.histogram(obj.headAngVel, bins=angEdges)
            headAngularDist[0:-1, idx] = counts * 100 / (sum(counts))

        """ make plots"""
        """distance plot"""
        WTBoot = bootstrap(distanceMat[:, self.WTIdx], 1,
                               distanceMat[:, self.WTIdx].shape[0])
        MutBoot = bootstrap(distanceMat[:, self.MutIdx], 1,
                                distanceMat[:, self.MutIdx].shape[0])
        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)

        distPlot = StartPlots()
        distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
                                     WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='Mut')
        distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
                                     MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        distPlot.ax.set_xlabel('Time (s)')
        distPlot.ax.set_ylabel('Distance travelled (px)')
        distPlot.legend(['WT', 'Mut'])
        # save the plot
        distPlot.save_plot('Distance traveled.tif', 'tif', savefigpath)
        distPlot.save_plot('Distance traveled.svg', 'svg', savefigpath)

        """velocity plot"""
        WTBoot = bootstrap(velocityDist[:, self.WTIdx], 1,
                               velocityDist[:, self.WTIdx].shape[0])
        MutBoot = bootstrap(velocityDist[:, self.MutIdx], 1,
                                velocityDist[:, self.MutIdx].shape[0])
        velPlot = StartPlots()
        velPlot.ax.plot(velEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        velPlot.ax.fill_between(velEdges, WTBoot['bootLow'],
                                    WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        velPlot.ax.plot(velEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        velPlot.ax.fill_between(velEdges, MutBoot['bootLow'],
                                    MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        velPlot.ax.set_xlabel('Velocity (px/s)')
        velPlot.ax.set_ylabel('Velocity distribution (%)')
        velPlot.legend(['WT', 'Mut'])
        velPlot.save_plot('Velocity distribution.tif', 'tif', savefigpath)
        velPlot.save_plot('Velocity distribution.svg', 'svg', savefigpath)

        """ plot angular velocity distribution"""
        WTBoot = bootstrap(angularDist[:, self.WTIdx], 1,
                               angularDist[:, self.WTIdx].shape[0])
        MutBoot = bootstrap(angularDist[:, self.MutIdx], 1,
                                angularDist[:, self.MutIdx].shape[0])

        """angular velocity plot"""
        angPlot = StartPlots()
        angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
                                    WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
                                    MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        angPlot.ax.set_xlabel('Angular velocity (radian/s)')
        angPlot.ax.set_ylabel('Angular velocity distribution (%)')
        angPlot.legend(['WT', 'Mut'])
        angPlot.save_plot('Angular velocity distribution.tif', 'tif', savefigpath)
        angPlot.save_plot('Angular velocity distribution.svg', 'svg', savefigpath)

        """plot head angular velocity distribution"""
        WTBoot = bootstrap(headAngularDist[:, self.WTIdx], 1,
                               headAngularDist[:, self.WTIdx].shape[0])
        MutBoot = bootstrap(headAngularDist[:, self.MutIdx], 1,
                                headAngularDist[:, self.MutIdx].shape[0])

        angPlot = StartPlots()
        angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
                                    WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
                                    MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        angPlot.ax.set_xlabel('Angular velocity(head) (radian/s)')
        angPlot.ax.set_ylabel('Angular velocity(head) distribution (%)')
        angPlot.legend(['WT', 'Mut'])
        angPlot.save_plot('Angular velocity(head) distribution.tif', 'tif', savefigpath)
        angPlot.save_plot('Angular velocity(head distribution.svg', 'svg', savefigpath)

        plt.close('all')

    def center_analysis(self, savefigpath):
        centerMat = np.full((self.minFrames, self.nSubjects), np.nan)
        plotT = np.arange(self.minFrames)/self.fps
        for idx, obj in enumerate(self.data['DLC_obj']):
            savefigFolder = os.path.join(self.analysisFolder, self.animals[idx])
            if not os.path.exists(savefigFolder):
                os.makedirs(savefigFolder)
            obj.moving_trace(savefigFolder)
            obj.get_time_in_center()
            centerMat[:,idx] = obj.cumu_time_center[0:self.minFrames]

        WTBoot = bootstrap(centerMat[:, self.WTIdx], 1,
                               centerMat[:, self.WTIdx].shape[0])
        MutBoot = bootstrap(centerMat[:, self.MutIdx], 1,
                                centerMat[:, self.MutIdx].shape[0])
        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)

        distPlot = StartPlots()
        distPlot.ax.plot(plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
                                     WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        distPlot.ax.plot(plotT, MutBoot['bootAve'], color=MutColor, label='Mut')
        distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
                                     MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        distPlot.ax.set_xlabel('Time (s)')
        distPlot.ax.set_ylabel('Time spent in the center (s)')
        distPlot.legend(['WT', 'Mut'])
        # save the plot
        distPlot.save_plot('Time spent in the center.tif', 'tif', savefigFolder)
        distPlot.save_plot('Time spent in the center.svg', 'svg', savefigFolder)

if __name__ == "__main__":
    root_dir = r'D:\openfield_cntnap'
    dataFolder = os.path.join(root_dir,'Data')
    animals = ['M1595', 'M1603', 'M1612', 'M1613', 'M1615', 'M1619']
    # add animal identity
    GeneBG = ['Mut', 'WT', 'WT', 'Mut', 'WT', 'Mut']
    fps = 40

    """analysis
    1. moving distance
    2. running speed distribution/average
    3. angular speed distribution/average
    4. time spend in the middle (future). A gui defining boudaries of the field based on user input
     """

    """ distance/speed related analysis"""

    #
    # make these plot functions?
    # plot the cumulative distance traveled
    savemotionpath = r'D:\openfield_cntnap\Summary\DLC'

    DLCSum = DLCSummary(dataFolder, animals, GeneBG, fps)

    # basic motor-related analysis
    #DLCSum.motion_analysis(savemotionpath)
    #DLCSum.center_analysis(savemotionpath)
    # analysis to do
    # moving trace ( require user input to mark the open field area)
    # time spend in the center (same)
    # tail movement in egocentric coordinates
    # savefigpath = r'D:\openfield_cntnap\Analysis\DLC\1595'
    # tt=DLCSum.data['DLC_obj'][0].moving_trace(savefigpath)
    """ keypoint-moseq analysis"""
    # DLCSum.data['DLC_obj'][0].get_time_in_center()
    moseqResults = r'D:\openfield_cntnap\Data\Moseq\results.h5'
    MoseqData = Moseq(root_dir)

    session = MoseqData.sessions[0]
    fps = 40
    savefigpath = os.path.join(root_dir, 'Analysis', 'DLC')

    #MoseqData.get_syllables(DLCSum)
    #MoseqData.load_syllable_plot(root_dir)
    MoseqData.tail_dist(DLCSum, savefigpath)

    #MoseqData.get_next_syllable()
    savefigpath = os.path.join(root_dir,'Analysis','Moseq')
    #MoseqData.syllable_frequency(DLCSum, savefigpath)
    #MoseqData.syllable_transition(DLCSum, savefigpath)
    for idx, session in enumerate(MoseqData.sessions):
        dlcObj = DLCSum.data['DLC_obj'][idx]
        videopath = DLCSum.data['Video'][idx]

        partLength = 60*dlcObj.fps  # save videos into 1 min files
        num_files = dlcObj.nFrames//partLength
        for nf in range(num_files):
            savevideopath = 'D:\\openfield_cntnap\\Analysis\\Video\\'+session[0:5]+'_'+str(nf)+'.mp4'
            print("Generating video #"+str(nf))
            if nf == num_files-1:
                frameRange = list(np.arange(nf * partLength, dlcObj.nFrames))
            else:
                frameRange = list(np.arange(nf * partLength, (nf + 1) * partLength))

            if not os.path.exists(savevideopath):
                MoseqData.multiprocess_videos(0.2, session, frameRange, dlcObj, videopath, savevideopath)
            else:
                print("video already exists")

    # moseq data:
    # syllable frequency
    # syllable transition matrix

    # 2. calculate moving speed using centroid, compare with speed of DLC
    MoseqVel = MoseqData.get_velocity(session, fps)
    MoseqData.syllable_analysis(session, fps)
    # 3. syllable appearance (total time of syllables)
    # syllable duration distribution (plotted for individual syllables
    # transition matrix
    x = 1

    # compare between sessions

# read csv file

# To do
# DLC:
#   1. try other body parts for velocity/distance
#   3. tail distribution (in moseq)

# Moseq:
#   1. clean the syllables
#   2. transition
#   3. average frequency/duration