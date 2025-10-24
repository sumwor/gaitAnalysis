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
#todo: to be added in Deeplabcut:
# 1. stride length (two hind paws at least)
# 2. step length
# 3. step width
# 4. nose/tail displacement
# Moseq:
# state transition matrix with time (darker color for current syllable, degraded color for
# previous syllables


""" questions for openfield behavior
1. individual variance v.s. genotype variance"""
import csv
import copy
import os.path
import ast
import pandas as pd
import glob
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
import imageio
from natsort import natsorted
import scipy
from scipy.signal import spectrogram,hilbert,correlate, find_peaks
from scipy.io import loadmat
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.stats import pearsonr
#import fitz
#from PIL import Image

from tqdm import tqdm
from pyPlotHW import *

from utils import *
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from tqdm import tqdm
from utility_HW import *
import h5py
import statsmodels.api as sm
from statsmodels.formula.api import ols, smf
import io
import subprocess as sp
import multiprocessing
import concurrent.futures
import functools
import seaborn as sns
import ruptures as rpt

# todo:
# 1. check the open field paper for related plots

class DLCData:

    def __init__(self, filePath, videoPath, rodspeedPath,analysisPath, fps):
        """

        :param filePath: DLC csv path
        :param videoPath:
        :param rodspeedPath: rod speed csv path
        :param analysisPath:
        :param fps: number (frames per second); path: file path with time stamps
        """

        self.filePath = filePath
        self.videoPath = videoPath
        self.rodPath = rodspeedPath
        self.nFrames = 0
        if fps.isnumeric():
            self.fps = fps
        else:
            # load the timeStamp csv
            time_raw = pd.read_csv(fps, header=None)
            self.t = np.array(time_raw[0]-time_raw[0][0])/1000
            self.t_start = time_raw[0][0]
        # read data
        self.data = self.read_data()
        self.analysis = analysisPath
        self.fieldSize = 40 # in centimeter, used to convert px to cm
        if not os.path.exists(self.analysis):
            os.makedirs(self.analysis)
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
            savedatapath = os.path.join(savefigpath, 'arena_coordinates.csv')
            if not os.path.exists(savedatapath):
                self.arena = frame_input(self.videoPath)
                # save the results:
                with open(savedatapath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['upper left',
                                     'upper right',
                                     'lower right',
                                     'lower left'])
                    writer.writerow([self.arena['upper left'],
                                    self.arena['upper right'],
                                    self.arena['lower right'],
                                    self.arena['lower left']])
                    f.close()
            else:
                # read data from file
                tempdata= pd.read_csv(savedatapath)
                self.arena = {}
                for key in tempdata.keys():
                    self.arena[key] = ast.literal_eval(tempdata[key].values[0])

                # convert px to cm
                # calculate the length of each side in pixels, get the average, then convert to cm
        sideLength = []
        arenaKeys = list(self.arena.keys())
        for kidx in range(len(arenaKeys)):
            key1 = arenaKeys[kidx]
            if kidx < len(arenaKeys)-1:
                key2 = arenaKeys[kidx + 1]
            else:
                key2 = arenaKeys[0]
            sideLength.append(np.sqrt((self.arena[key1][0]-self.arena[key2][0])**2+
                                              (self.arena[key1][1]-self.arena[key2][1])**2))
        # save the coordinates in analysis folder
        self.px2cm = self.fieldSize/np.mean(sideLength)

        arena_x = [self.arena['upper left'][0], self.arena['upper right'][0],
                   self.arena['lower right'][0], self.arena['lower left'][0], self.arena['upper left'][0]]
        arena_y = [self.arena['upper left'][1], self.arena['upper right'][1],
                   self.arena['lower right'][1], self.arena['lower left'][1], self.arena['upper left'][1]]

        # get the instantaneous distance from center
        slope1 = (self.arena['upper left'][1] - self.arena['lower right'][1]) / (self.arena['upper left'][0] - self.arena['lower right'][0])
        slope2 = (self.arena['lower left'][1] - self.arena['upper right'][1]) / (self.arena['lower left'][0] - self.arena['upper right'][0])

        # Calculate the x-coordinate of the intersection point
        x_intersection = ((self.arena['upper right'][1] - self.arena['upper left'][1]) + slope1 * self.arena['upper left'][0] - slope2 * self.arena['upper right'][0]) / (slope1 - slope2)

        # Calculate the y-coordinate of the intersection point
        y_intersection = slope1 * (x_intersection - self.arena['upper left'][0]) + self.arena['upper left'][1]

        self.center_point = [x_intersection, y_intersection]

        self.dist_center = np.sqrt((np.array(self.data['tail 1']['x'])-self.center_point[0])**2 +
                                   (np.array(self.data['tail 1']['y'])-self.center_point[1])**2)

        if hasattr(self, 'px2cm'):
            self.dist_center = self.dist_center*self.px2cm

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
            num_cross = 0
            self.num_cross = []  # number of times the animal crosses the border line of center area
            for idx in range(len(self.data['tail 1']['x'])):
                if self.data['tail 1']['x'][idx] > x_left and self.data['tail 1']['x'][idx] < x_right:
                    if self.data['tail 1']['y'][idx] > y_upper and self.data['tail 1']['y'][idx] < y_lower:
                        is_center[idx] = 1

                        if idx > 0:
                            if is_center[idx] != is_center[idx-1]:
                                num_cross+=1
                self.num_cross.append(num_cross)

            self.time_in_center = is_center
            self.cumu_time_center = []
            cumu = 0
            for f in range(self.nFrames):
                cumu += self.time_in_center[f]/self.fps
                self.cumu_time_center.append(cumu)

        else:
            print("please run moving_trace first")

    def plot_distance_to_center(self, t, savefigpath):
        # plot the distribution of distance to center
        # as well as a function of time
        distPlot = StartPlots()
        self.dist_center_bins = distPlot.ax.hist(self.dist_center, bins = np.linspace(0, 1200, 101))
        self.dist_center_bins_30 = distPlot.ax.hist(self.dist_center[0:30*60*self.fps],
                                                    bins = np.linspace(0,1200, 101))
        distPlot.ax.set_xlabel('Distance from center (px)')
        distPlot.ax.set_ylabel('Occurance')
        distPlot.save_plot('Distribution of distance from center.tiff', 'tiff', savefigpath)
        # average distance from center in a running window
        self.dist_center_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
        for ff in range(self.nFrames - 1 - t*self.fps):
            self.dist_center_running[ff] = np.nanmean(self.dist_center[ff:ff+t*self.fps])

        distRunningPlot = StartPlots()
        distRunningPlot.ax.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.dist_center_running)
        distRunningPlot.ax.set_ylabel('Average distance from center (px)')
        distRunningPlot.ax.set_xlabel('Time (s)')

        distRunningPlot.save_plot('Average distance from center.tiff', 'tiff', savefigpath)
        plt.close('all')

    def read_data(self):
        data = {}
        if not hasattr(self, 't'):
            self.t = []

        if isinstance(self.filePath, str):
            with open(self.filePath) as csv_file:
                print("Loading data from: " + self.filePath)
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
                        #self.t.append(0)
                        line_count += 1
                        self.nFrames += 1

                    else:
                        tempList = ['x', 'y', 'p']
                        for ii in range(len(row) - 1):
                            # get the corresponding body parts based on index
                            body = data['bodyparts'][int(np.floor((ii) / 3))]
                            data[body][tempList[np.mod(ii, 3)]].append(float(row[ii + 1]))
                        #self.t.append(self.nFrames*(1/self.fps))
                        line_count += 1
                        self.nFrames += 1

                print(f'Processed {line_count} lines.')

                # add frame time
                #tStep= 1/self.fps
                data['time'] = self.t
                #self.t = np.array(self.t)
        else:
            data['time'] = np.nan
        # load rod speed data
        rodSpeed = pd.read_csv(self.rodPath, header=None)
        data['rodSpeed'] = rodSpeed.iloc[:, 0].values
        data['rodT'] = (rodSpeed.iloc[:, 1].values-self.t_start)/1000

        #%% for estimations with likelihood less than 0.8, replace the value with linear fit
        # based on previous and next value
        # corrected_data=copy.deepcopy(data)
        # kp_list = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot',
        #            'nose', 'left ear', 'right ear','left hand', 'right hand']
        # corrected_frames = []
        # for kp in kp_list:
        #     data[kp]['x'] = np.array(data[kp]['x'])
        #     data[kp]['y'] = np.array(data[kp]['y'])
        #     data[kp]['p'] = np.array(data[kp]['p'])
        #     for i in range(6, len(data['time'])-6):
        #         if data[kp]['p'][i] < 0.8:
        #             prev_reliable={}
        #             next_reliable = {}
        #             prev_reliable['x'] = data[kp]['x'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             prev_reliable['y'] = data[kp]['y'][max(0, i - 5):i][data[kp]['p'][max(0, i - 5):i] >= 0.8]
        #             next_reliable['x'] = data[kp]['x'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             next_reliable['y'] = data[kp]['y'][i + 1:min(len(data[kp]['p']), i + 6)][data[kp]['p'][i + 1:min(len(data[kp]['p']), i + 6)] >= 0.8]
        #             prev_reliable = pd.DataFrame(prev_reliable)
        #             next_reliable = pd.DataFrame(next_reliable)
        #             reliable_points = pd.concat([prev_reliable, next_reliable])
        #
        #             # If we found any reliable points, replace the unreliable x and y values
        #             if not reliable_points.empty:
        #                 if not i in corrected_frames:
        #                     corrected_frames.append(i)
        #                 # Calculate average x and y of these reliable points
        #                 avg_x = reliable_points['x'].mean()
        #                 avg_y = reliable_points['y'].mean()
        #
        #                 # Replace unreliable x and y values with the interpolated average
        #                 corrected_data[kp]['x'][i] = avg_x
        #                 corrected_data[kp]['y'][i] = avg_y
        #
        # # plot some frames to examine it
        # frame_num = 10198
        # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
        # plt.figure()
        # plt.imshow(curr_frame)
        # cmap = cm.get_cmap('viridis', len(kp_list))
        # for kp in kp_list:
        #     plt.scatter(data[kp]['x'][frame_num], data[kp]['y'][frame_num], c=cmap(kp_list.index(kp)), s=200,label = kp)
        #     plt.scatter(corrected_data[kp]['x'][frame_num],corrected_data[kp]['y'][frame_num],marker = 'x',c=cmap(kp_list.index(kp)), s=200,label = kp+'_corrected')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return data

    def check_quality(self):
        # go through the data and plot the distribution of p-values
        pass

    def get_movement(self):
        # calculate distance, running velocity, acceleration, based on tail (base of tail)
        savedatapath = os.path.join(self.analysis,'movement.pickle')
        if not os.path.exists(savedatapath):
            self.vel = np.zeros((self.nFrames-1, 1))
            self.dist = np.zeros((self.nFrames-1,1))
            self.accel = np.zeros((self.nFrames-1, 1))

            for ff in range(self.nFrames-1):
                self.dist[ff] = np.sqrt((self.data['tail 1']['x'][ff+1] - self.data['tail 1']['x'][ff])**2 +
                    (self.data['tail 1']['y'][ff + 1] - self.data['tail 1']['y'][ff]) ** 2)

                self.vel[ff] = (self.dist[ff])*self.fps
                if ff<self.nFrames-2:
                    self.accel[ff] = (self.vel[ff+1]-self.vel[ff])*self.fps
            # save vel, dist, accel in pickle file
            dist = self.dist
            vel = self.vel
            accel = self.accel
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist, vel, accel], f)
            f.close()
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist, self.vel, self.accel = pickle.load(f)
            f.close()

        if hasattr(self, 'px2cm'):
            self.dist = self.dist*self.px2cm
            self.vel = self.vel*self.px2cm
            self.accel = self.accel*self.px2cm

    def get_movement_running(self, t, savefigpath):
        savedatapath = os.path.join(self.analysis, 'movement_running.pickle')
        if not os.path.exists(savedatapath):
        # get average distance and velocity in running window of t seconds
            self.vel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.dist_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.accel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.dist_running[ff] = np.sum(self.dist[ff:ff+t*self.fps])
                self.vel_running[ff] = np.nanmean(self.vel[ff:ff+t*self.fps])
                self.accel_running[ff] = np.nanmean(self.accel[ff:ff+t*self.fps])
                self.vel[ff] = (self.dist[ff]) * self.fps

            dist_running = self.dist_running
            vel_running = self.vel_running
            accel_running = self.accel_running
            with open(savedatapath, 'wb') as f:
                pickle.dump([dist_running, vel_running, accel_running], f)
            f.close()

            velPlot = StartPlots()
            velPlot.ax.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.dist_running)
            velPlot.ax.set_ylabel('Average distance traveled (px)')
            #ax2 = velPlot.ax.twinx()
            #ax2.plot(self.t[0:self.nFrames - 1 - t * self.fps], self.vel_running, color='red')
            #ax2.set_ylabel('Average velocity')
            velPlot.ax.set_xlabel('Time (s)')

            velPlot.save_plot('Running distance and velocity.png', 'png', savefigpath)
            plt.close(velPlot.fig)
        else:
            # load dis, vel and accel from pickle file
            with open(savedatapath, 'rb') as f:
                self.dist_running, self.vel_running, self.accel_running = pickle.load(f)
            f.close()
        if hasattr(self, 'px2cm'):
            self.dist_running = self.dist_running*self.px2cm
            self.vel_running= self.vel_running*self.px2cm
            self.accel_running = self.accel_running*self.px2cm

    def get_angular_velocity(self):
        # calculate angular velocity based on tail and spine 1
        savedatapath = os.path.join(self.analysis, 'angular_velocity.pickle')
        if not os.path.exists(savedatapath):
            self.angVel = np.zeros((self.nFrames-1, 1))
            for ff in range(self.nFrames-1):
                y1 = self.data['spine 1']['y'][ff] - self.data['tail 1']['y'][ff]
                x1 = self.data['spine 1']['x'][ff] - self.data['tail 1']['x'][ff]

                y2 = self.data['spine 1']['y'][ff+1] - self.data['tail 1']['y'][ff+1]
                x2 = self.data['spine 1']['x'][ff+1] - self.data['tail 1']['x'][ff+1]

                self.angVel[ff] = self.get_angle([x1, y1], [x2, y2])*self.fps
            angVel = self.angVel
            with open(savedatapath, 'wb') as f:
                pickle.dump(angVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel = pickle.load(f)
            f.close()
        #self.angVel = self.angVel*self.fps

    def get_head_angular_velocity(self):
        savedatapath = os.path.join(self.analysis, 'head_angular_velocity.pickle')
        if not os.path.exists(savedatapath):
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
            with open(savedatapath, 'wb') as f:
                pickle.dump(self.headAngVel, f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.headAngVel = pickle.load(f)
            f.close()

    def get_stride(self,front_kp, back_kp, df_entry):
        savedatapath = os.path.join(self.analysis, 'stride_freq.csv')
        runFile = os.path.join(self.analysis, 'notExist.csv') # a not existing file to allow re-calculating
        if not os.path.exists(runFile):

            savefigpath = os.path.join(self.analysis)
            if not os.path.exists(savefigpath):
                os.makedirs(savefigpath)

            # get rid the the turning period
            
            # %% define rod plane first
            # load reference point
            ave_left_rod_back = self.data['left_rod_back']
            ave_right_rod_back = self.data['right_rod_back']
            ave_center_rod_back = self.data['center_rod_back']
            ave_left_rod_front = self.data['left_rod_front']
            ave_right_rod_front = self.data['right_rod_front']
            ave_center_rod_front = self.data['center_rod_front']

            ref_plot = os.path.join(savefigpath, 'Rod coordinate.png')
            if not os.path.exists(ref_plot):
                frame = read_video(self.videoPath, 0, ifgray=False)
                # overlay the video frame?
                plt.figure()
                plt.imshow(frame)
                plt.scatter(self.data['rod_left_back']['x'], self.data['rod_left_back']['y'],
                            c=self.data['rod_left_back']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood
                plt.colorbar(label='Confidence')

                plt.scatter(self.data['rod_right_back']['x'], self.data['rod_right_back']['y'],
                            c=self.data['rod_right_back']['p'], cmap='viridis', s=100)

                plt.scatter(self.data['rod_left_front']['x'], self.data['rod_left_front']['y'],
                            c=self.data['rod_left_front']['p'], cmap='viridis', s=100)

                # Add color bar to show the scale of likelihood

                plt.scatter(self.data['rod_right_front']['x'], self.data['rod_right_front']['y'],
                            c=self.data['rod_right_front']['p'], cmap='viridis', s=100)

                # get average from keypoints with confidence higher than 95


                plt.scatter(ave_left_rod_back[0],ave_left_rod_back[1], s=500)
                plt.scatter(ave_right_rod_back[0], ave_right_rod_back[1], s=500)
                plt.scatter(ave_center_rod_back[0], ave_center_rod_back[1], s=500)

                plt.scatter(ave_left_rod_front[0],ave_left_rod_front[1], s=500)
                plt.scatter(ave_right_rod_front[0], ave_right_rod_front[1], s=500)
                plt.scatter(ave_center_rod_front[0], ave_center_rod_front[1], s=500)

                plt.savefig(os.path.join(savefigpath, 'rod_plane.png'))
                plt.close()
            # save the figure

            # %% examine the body parts in back view and front view

            # find behavior time (from rod start turning to fall)
            startTime= self.data['rodT'][self.data['rodSpeed_smoothed']>0][0]
            if np.isnan(self.data['rodStart'][0]):
                self.data['rodStart'][0] = 0
            endTime = startTime+df_entry['TimeOnRod'] + self.data['rodRun'][0] - self.data['rodStart'][0] # need the time stay on rod

            timeMaskDLC = np.logical_and(self.data['time']>=startTime, self.data['time']<= endTime)
            timeMaskRod = np.logical_and(self.data['rodT']>=startTime, self.data['rodT']<= endTime)
            nFramesInclude = np.sum(timeMaskDLC)
            time_include = self.data['time'][timeMaskDLC]
            kp_list = ['left hand', 'right hand', 'left foot', 'right foot']
            self.stride = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_front[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_front[1])**2)
                elif 'foot' in kp:
                    self.stride[:,idx] = np.sqrt((np.array(self.data[kp]['x'])[timeMaskDLC]-ave_center_rod_back[0])**2 +
                                            (np.array(self.data[kp]['y'])[timeMaskDLC]-ave_center_rod_back[1])**2)

            # try calculate the stride using distance from the rod (a horizontal line)
            self.stride_rod = np.full((nFramesInclude, len(kp_list)), np.nan)

            #dataMask = np.logical_and(timeMask, p_mask)
            #self.notnanChunks = {}  # save the indices of not nan chunks in the stride for later filtering
            for idx,kp in enumerate(kp_list):
                if 'hand' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_left_rod_front, ave_right_rod_front)
                elif 'foot' in kp:
                    self.stride_rod[:,idx] = distance_points_to_line(np.array(self.data[kp]['x'])[timeMaskDLC],
                                                                    np.array(self.data[kp]['y'])[timeMaskDLC],
                                                                    ave_right_rod_back, ave_left_rod_back)

            #tempMask = ~p_mask[timeMask]
            #    self.stride[tempMask,idx] = np.nan
            #    tempStride, tempIdx = fill_nans_and_split(self.stride[:, idx])
                # interpolate the nans
                # self.notnanChunks[kp] = tempIdx
                # for ich, chunk in enumerate(tempIdx):
                #     self.stride[chunk[0]:chunk[1]+1,idx] = tempStride[ich]


            # low-pass filter the data
            fps = 50
            self.t_interp = np.arange(time_include[0], time_include[-1] + 1 / fps, 1 / fps)
            self.filtered_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)
            self.interp_stride = np.full((len(self.t_interp), len(kp_list)), np.nan)

            # need to determine the cutoff frequency here
            for idx, kp in enumerate(kp_list):
                #for ich, chunk in enumerate(self.notnanChunks[kp]):
                #    if chunk[1]-chunk[0]+1 > 18:  # padlen
                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
                self.interp_stride[:,idx] = np.interp(self.t_interp, time_include, self.stride_rod[:,idx])

                self.filtered_stride[:,idx] = butter_lowpass_filter(self.interp_stride[:,idx], 5,fps,order=5)

            #%%
            # examine the autocorrelation
            # average them over genotype and trial
            # find the time when rod speed reach 5/10
            #if df_entry['Trial']<=6:
            startSpeed = 5
            #else:
            #    startSpeed = 10
            startTime_auto = self.data['rodT'][self.data['rodSpeed_smoothed']>startSpeed][0]
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for 4 subplots
            ax = ax.flatten()
            for ss in range(len(kp_list)):
                signal = pd.Series(self.filtered_stride[self.t_interp>startTime_auto,ss])
                autocorr_values = [signal.autocorr(lag=i) for i in range(len(signal)//2)]

                plot_time = 10
                # Subplot 1 (First row, spanning two columns)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, autocorr_values, linewidth=0.5)
                ax[ss].plot(np.arange(len(autocorr_values))/fps, np.zeros(len(autocorr_values)),c='r', linewidth=2)
                #ax[ss].stem(range(len(autocorr_values)), autocorr_values,linefmt='b-', basefmt=" ", use_line_collection=True)
                ax[ss].set_title('Autocorrelation of ' + kp_list[ss])

                if ss==0:
                    # save autocorrelation value and lags in dataframe
                    autocorr_df = pd.DataFrame({'lags': np.arange(len(autocorr_values))/fps})
                autocorr_df[kp_list[ss]] = autocorr_values

            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis, 'Stride autocorrelation.png'), dpi=300)  # Save as PNG fil
            # save autocorrelation
            autocorr_df.to_csv(os.path.join(self.analysis, 'Stride autocorrelation.csv'))
            plt.close()

            #%%
            # instantaneous frequency with hilbert transform

            # Compute the analytic signal
            #
            # analytic_signal = hilbert(self.interp_stride[:,2])
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (1 / fps))


            # Plot spectrogram

            # %% short time fourier transform
            # from scipy.signal import stft
            # frequencies, times, Zxx = stft(self.filtered_stride[:,3], fs=50, nperseg=256)
            # plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
            # plt.colorbar(label='Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [s]')
            # plt.title('STFT Magnitude')

            #%% pearson correlation between limbs
            # phase lag?
            # generate some plots
            pcorr = pd.DataFrame({'time': self.t_interp})
            corr_group = [['left hand','right hand'], ['left foot', 'right foot'],
                           ['left hand', 'left foot'], ['right hand', 'right foot']]
            corr_Idx = [[0,1], [2,3], [0,2], [1,3]]
            # xcorr between hands/feet/left/right
            timeStep = 2 # in second
            for kp_pairs,kp_idx in zip(corr_group,corr_Idx):
                corrCoeff_running = np.zeros((len(self.t_interp)))
                for idx,t in enumerate(self.t_interp):
                    tMask = np.logical_and(self.t_interp>t, self.t_interp <t+timeStep)
                    corrCoeff_running[idx] = np.corrcoef(self.filtered_stride[tMask,kp_idx[0]],
                                                             self.filtered_stride[tMask,kp_idx[1]])[0,1]
                pcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corrCoeff_running
            
            # cross correlation
            max_lag_sec = 1.0  # maximum lag to compute (in seconds)
            dt = self.t_interp[1] - self.t_interp[0]  # time step of your signal
            max_lag_samples = int(max_lag_sec / dt)

            # Store results
            max_xcorr = pd.DataFrame({'time': self.t_interp})
            max_lag = pd.DataFrame({'time': self.t_interp})

            for kp_pairs, kp_idx in zip(corr_group, corr_Idx):
                # Each element will be a 2D array: shape (len(t_interp), 2*max_lag_samples+1)
                # Arrays for max correlation and lag at each time point
                corr_max = np.full(len(self.t_interp), np.nan)
                lag_at_max = np.full(len(self.t_interp), np.nan)

                lags = np.arange(-max_lag_samples, max_lag_samples + 1) * dt

                for idx, t in enumerate(self.t_interp):
                    # 2-second window mask
                    tMask = (self.t_interp > t) & (self.t_interp < t + timeStep)
                    x = self.filtered_stride[tMask, kp_idx[0]]
                    y = self.filtered_stride[tMask, kp_idx[1]]

                    if len(x) < 2 or len(y) < 2:
                        continue

                    # Normalize signals
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Compute normalized cross-correlation
                    c = correlate(y, x, mode='full')
                    c = c / (np.std(x) * np.std(y) * len(x))

                    # Center index
                    mid = len(c) // 2
                    c_window = c[mid - max_lag_samples: mid + max_lag_samples + 1]

                    # Find max correlation and corresponding lag
                    max_idx = np.nanargmax(c_window)
                    corr_max[idx] = c_window[max_idx]
                    lag_at_max[idx] = lags[max_idx]
                
                max_xcorr[kp_pairs[0]+'-'+kp_pairs[1]] = corr_max
                max_lag[kp_pairs[0]+'-'+kp_pairs[1]] = lag_at_max

            #save cross correlation results
            pcorr_renamed = pcorr.copy()
            pcorr_renamed.columns = ['time'] + [col + '_pearson' for col in pcorr.columns[1:]]

            max_xcorr_renamed = max_xcorr.copy()
            max_xcorr_renamed.columns = ['time'] + [col + '_maxxcorr' for col in max_xcorr.columns[1:]]

            max_lag_renamed = max_lag.copy()
            max_lag_renamed.columns = ['time'] + [col + '_lag' for col in max_lag.columns[1:]]

            # 2. Merge all DataFrames on 'time'
            combined_df = pcorr_renamed.merge(max_xcorr_renamed, on='time').merge(max_lag_renamed, on='time')

            # 3. Save to CSV
            combined_df.to_csv(os.path.join(self.analysis, 'Stride correlation.csv'), index=False)

            # make a plot to show pearson correlation and cross correlation and max lag
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-right hand'], label= 'Hands')
            ax[1].plot(self.t_interp, pcorr['left foot-right foot'], label = 'Feet')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-right hand'], label= 'Hands')
            ax[2].plot(self.t_interp, max_xcorr['left foot-right foot'], label = 'Feet')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-right hand'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['left foot-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - HF.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            # same plot to show left and right
            fig,ax = plt.subplots(4, 1, figsize=(16, 10))
            # Subplot 1 (First row, spanning two columns)
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)
            #ax[0].plot(self.t_interp , self.filtered_stride[:,1])
            #ax[0].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            
            # plot pearson correlation of hands and foot
            ax[1].plot(self.t_interp, pcorr['left hand-left foot'], label= 'Left')
            ax[1].plot(self.t_interp, pcorr['right hand-right foot'], label = 'Right')
            #ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Pearson correlation coefficient')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # plot cross correlation 
            ax[2].plot(self.t_interp, max_xcorr['left hand-left foot'], label= 'LEft')
            ax[2].plot(self.t_interp, max_xcorr['right hand-right foot'], label = 'Right')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)
            ax[2].set_title('Max cross correlation coefficient')
            #ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

            # plot max lag
            ax[3].plot(self.t_interp, max_lag['left hand-left foot'], label= 'Hands')
            ax[3].plot(self.t_interp, max_lag['right hand-right foot'], label = 'Feet')
            ax[3].set_title('Max lag (s)')
            ax[3].legend(loc='upper left', bbox_to_anchor=(1, 1))

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            plt.savefig(os.path.join(self.analysis,'Stride correlation - LR.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            #%% calculate hand/foot step amplitude and frequency based on peak detection
            self.stride_amp = []
            self.stride_time = []
            self.stride_freq = np.full(self.filtered_stride.shape, np.nan)

            time = self.t_interp
            for ll in range(4): # step size and amplitude of 4 limbs
            # Detect peaks (foot lifts)
            
                distance = self.filtered_stride[:,ll]
                peaks, props = find_peaks(distance, prominence=2, distance=None)

                # Estimate baseline before each step using local minima
                inv_distance = -distance
                troughs, _ = find_peaks(inv_distance, prominence=2 / 2, distance=None)

                step_amplitudes = []
                step_times = []

                for peak in peaks:
                    # Find the closest following trough (baseline)
                    next_troughs = troughs[troughs > peak]
                    if len(next_troughs) == 0:
                        continue
                    baseline_idx = next_troughs[0]
                    amplitude = distance[peak] - distance[baseline_idx]
                    step_amplitudes.append(amplitude)
                    step_times.append(time[peak])

                step_amplitudes = np.array(step_amplitudes)
                step_times = np.array(step_times)

                self.stride_amp.append(step_amplitudes)
                self.stride_time.append(step_times)


                # Compute step frequency (Hz) in running 2 second window
                window = 2
                freqs = np.full(len(time), np.nan)  # preallocate

                for i, t in enumerate(time):
                    # count steps within [t - window/2, t + window/2]
                    mask = (step_times >= t - window/2) & (step_times <= t + window/2)
                    steps_in_window = step_times[mask]

                    if len(steps_in_window) >= 1:
                        intervals = np.diff(steps_in_window)
                        freqs[i] = 1 / np.mean(intervals)
                    else:
                        freqs[i] = np.nan

                self.stride_freq[:,ll] = freqs


            fig,ax = plt.subplots(5, 1, figsize=(16, 16))
            # rod speed
            ax[0].plot(self.data['rodT'],self.data['rodSpeed_smoothed'])
            for start_idx, end_idx in self.data['turning_period']:
                ax[0].axvspan(self.data['time'][start_idx], self.data['time'][end_idx],
                    color='grey', alpha=0.3)
            ax[0].set_title('Rod speed')
            ax[0].set_ylabel('Rod speed (RPM)')
            ax[0].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 2, stride of hand
            ax[1].plot(self.t_interp, self.filtered_stride[:,0])
            ax[1].plot(self.t_interp , self.filtered_stride[:,1])
            ax[1].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[1].set_title('Distance between left/right hand and the rod')
            ax[1].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 3 foot
            ax[2].plot(self.t_interp, self.filtered_stride[:,2])
            ax[2].plot(self.t_interp, self.filtered_stride[:,3])
            ax[2].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].set_title('Distance between left/right foot and the rod')
            ax[2].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 hand amplitude
            ax[3].stem(self.stride_time[0], self.stride_amp[0], linefmt='C0-',  basefmt=" ", label='left hand')
            ax[3].stem(self.stride_time[1], self.stride_amp[1],linefmt='C1-',  basefmt=" ",label='right hand')
            ax[3].legend(['left hand', 'right hand'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[3].set_title('Hand step amplitude')
            ax[3].tick_params(axis='x', which='both', labelbottom=False)

            # Subplot 4 (Third row, first column)
            ax[4].stem(self.stride_time[2], self.stride_amp[2], linefmt='C0-',  basefmt=" ", label='left foot')
            ax[4].stem(self.stride_time[3], self.stride_amp[3], linefmt='C1-',  basefmt=" ", label='right foot')
            ax[4].legend(['left foot', 'right foot'],loc='upper left', bbox_to_anchor=(1, 1))
            ax[4].set_title('Foot step amplitude')

            for a in ax:  # ax is a list/array of subplots
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
            
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis,'Stride amplitude.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            data = {'left hand': self.filtered_stride[:,0],
                    'right hand': self.filtered_stride[:,1],
                    'left foot': self.filtered_stride[:, 2],
                    'right foot': self.filtered_stride[:, 3],
                    'stride amplitude': self.stride_amp,
                    'stride time': self.stride_time,
                    'stride frequency': self.stride_freq,
                    'time': self.t_interp}
            #dataDF = pd.DataFrame(data)
            #dataDF.to_csv(savedatapath)
            # save to pickle file
            with open( os.path.join(self.analysis, 'stride_freq.pickle'), 'wb') as f:
                pickle.dump(data, f)

            #%%
            # cumulative area under the curve
            # cum_xcorr_foot = np.cumsum(xcorr['left foot-right foot'])/fps
            # cum_xcorr_hand = np.cumsum(xcorr['left hand-right hand']) / fps
            # cum_xcorr_left = np.cumsum(xcorr['left hand-left foot'])/fps
            # cum_xcorr_right = np.cumsum(xcorr['right hand-right foot']) / fps
            # plt.figure()
            # plt.plot(self.t_interp, cum_xcorr_foot)
            # plt.plot(self.t_interp, cum_xcorr_hand)
            # plt.plot(self.t_interp, cum_xcorr_left)
            # plt.plot(self.t_interp, cum_xcorr_right)
            # plt.plot(self.data['rodT'][timeMaskRod], self.data['rodSpeed_smoothed'][timeMaskRod])
            # plt.xlabel('time')
            # plt.ylabel('Cumulative area under the curve of xcorr')
            # plt.legend(['feet','hands','left','right', 'Rod speed'])
            # plt.savefig(os.path.join(self.analysis,'Stride correlation.png'), dpi=300)  # Save as PNG fil
            # #plt.show()
            # plt.close()
            # # cross correlation in 10 second window

            # # save data in csv
            # xcorr.to_csv(os.path.join(self.analysis, 'Stride crosscorrelation.csv'))

            #
            #%% tail angle
            # calculate spine 3 - tail 1 - tail 2 angle
            A = np.array([self.data['spine 3']['x'], self.data['spine 3']['y']]).T
            B = np.array([self.data['tail 1']['x'], self.data['tail 1']['y']]).T
            C = np.array([self.data['tail 2']['x'], self.data['tail 2']['y']]).T

            A = A[timeMaskDLC,:]
            B = B[timeMaskDLC,:]
            C = C[timeMaskDLC,:]
            # Calculate vectors AB and BC
            AB = B - A
            BC = C - B

            # Calculate the angle between AB and BC
            # Calculate dot and cross products for each time point
            dot_product = np.sum(AB * BC, axis=1)  # Dot product for each row (time point)
            cross_product = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]  # Cross product for each time point

            # Calculate the angle at each time point
            angles = np.arctan2(cross_product, dot_product)

            # Convert to degrees
            angles = np.degrees(angles)

            # interpolate and filter the angle
            fps = 50
            self.filtered_angle = np.full((len(self.t_interp)), np.nan)
            self.interp_angle = np.full((len(self.t_interp)), np.nan)

            # need to determine the cutoff frequency here

                # interpolate the data first. Original data were recorded with unstable fps. (around 50)
            self.interp_angle= np.interp(self.t_interp, time_include, angles)

            self.filtered_angle = butter_lowpass_filter(self.interp_angle, 5,fps,order=5)


            # save data in csv
            tail_angle= pd.DataFrame({'angle':self.filtered_angle, 'time':self.t_interp})
            tail_angle.to_csv(os.path.join(self.analysis, 'Tail angle.csv'))
            # Calculate the angle in radians using atan2 for correct sign

            # plot the video frame with keypoint estimatino
            # frame_num = 7760
            # curr_frame = read_video(self.videoPath, frame_num, ifgray=False)
            # plt.figure()
            # plt.imshow(curr_frame)
            # kp_plot = ['tail 2']
            # for kp in kp_plot:
            #     plt.scatter(self.data[kp]['x'][frame_num], self.data[kp]['y'][frame_num], s=20)

            #%% head angle

            #%% tail position
            # plot the density distribution of the tail
            # set coordinate of tail 1 to be (0, 0)
            # ego_tail = {}
            # tail_key = ['tail 1', 'tail 2', 'tail 3']
            # for t in tail_key:
            #     ego_tail[t] = {}
            #     ego_tail[t]['x']= np.array(self.data[t]['x'])-np.array(self.data['tail 1']['x'])
            #     ego_tail[t]['y'] = np.array(self.data[t]['y']) - np.array(self.data['tail 1']['y'])
            #
            # plt.figure(figsize=(12, 6))
            # # Density plot for aligned b coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 2']), x='x', y='y',
            #             fill=True, cmap='Blues', alpha=0.5, label='Point B',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm())
            # # Density plot for aligned c coordinates
            # sns.kdeplot(data=pd.DataFrame(ego_tail['tail 3']), x='x', y='y',
            #             fill=True, cmap='Reds', alpha=0.5, label='Point C',
            #             thresh=0.001,  # Avoid clipping at 0
            #             levels=20,
            #             norm=LogNorm()
            #             )
            # plt.axhline(0, color='black', lw=1, ls='--', label='y = 0')
            # plt.axvline(0, color='black', lw=1, ls='--', label='x = 0')
            #
            # plt.title('Density Distribution of Aligned Points B and C')
            # plt.xlabel('Aligned B X')
            # plt.ylabel('Aligned B Y / Aligned C Y')
            # plt.axhline(0, color='black', lw=0.5, ls='--')
            # plt.axvline(0, color='black', lw=0.5, ls='--')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # with open(savedatapath, 'wb') as f:
            #     pickle.dump(self.stride, self. f)
            # f.close()
        else:
            print("Analysis already done")
            return np.nan

    def get_result(self):
        # go over the behavior result and get time before fell
        x=1
        rodData = read_rotarod_csv()
    def get_angular_velocity_running(self, t, savefigpath):
        # calculate angular velocity with running window t
        savedatapath = os.path.join(self.analysis, 'angular_velocity_running.pickle')

        if not os.path.exists(savedatapath):
            self.angVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))
            self.headAngVel_running = np.zeros((self.nFrames - 1 - t*self.fps, 1))

            for ff in range(self.nFrames - 1 - t*self.fps):
                self.angVel_running[ff] = np.nanmean(self.angVel[ff:ff+t*self.fps])
                self.headAngVel_running[ff] = np.nanmean(self.headAngVel[ff:ff+t*self.fps])

            # plot the velocity here
            angPlot = StartPlots()
            angPlot.ax.plot(self.t[0:self.nFrames - 1 - t*self.fps], self.angVel_running)
            angPlot.ax.set_ylabel('Angular velocity')
            ax2 = angPlot.ax.twinx()
            ax2.plot(self.t[0:self.nFrames - 1 - t*self.fps], self.headAngVel_running, color='red')
            ax2.set_ylabel('Head angular velocity', color='red')
            angPlot.ax.set_xlabel('Time (s)')

            angPlot.save_plot('Running angular vel.tiff', 'tiff', savefigpath)
            plt.close(angPlot.fig)

            angVel_running = self.angVel_running
            headAngVel_running = self.headAngVel_running
            with open (savedatapath, 'wb') as f:
                pickle.dump([angVel_running,headAngVel_running], f)
            f.close()
        else:
            with open(savedatapath, 'rb') as f:
                self.angVel_running, self.headAngVel_running = pickle.load(f)
            f.close()

    def get_angle(self, v1, v2):
        # get angle between two vectors
            v1_u = self.unit_vector(v1)
            v2_u = self.unit_vector(v2)

            angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            if v1[0] * v2[1] - v1[1] * v2[0] < 0:
                angle = -angle
            return angle

    def plot_keypoints(self, nFrame):
        bodyparts = self.data['bodyparts']
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
        image = read_video(self.videoPath, nFrame, ifgray=True)
        plt.imshow(image)
        for bd in bodyparts:
            plt.scatter(self.data[bd]['x'][nFrame], self.data[bd]['y'][nFrame])
        for sk in skeleton:
            plt.plot([self.data[sk[0]]['x'][nFrame],self.data[sk[1]['x'][nFrame]]], [self.data[sk[0]]['y'][nFrame],self.data[sk[1]['y'][nFrame]]])

        plt.show()

    def unit_vector(self, v):
        """ Returns the unit vector of the vector.  """
        return v / np.linalg.norm(v)

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

    def __init__(self, root_folder, fps, groups,behavior):
        self.rootFolder = root_folder
        self.dataFolder = os.path.join(root_folder, 'Data')
        self.analysisFolder = os.path.join(root_folder, 'Analysis')
        self.sumFolder = os.path.join(root_folder, 'Summary')
        output = self.get_animalInfo(root_folder)
        self.animals = output['animals']
        self.GeneBG = output['geneBG']
        self.Sex = output['sex']
        self.fps = fps
        self.behavior = behavior
        self.nSessions = 0
        # make directories
        if not os.path.exists(self.analysisFolder):
            os.makedirs(self.analysisFolder)
        if not os.path.exists(self.sumFolder):
            os.makedirs(self.sumFolder)
        #self.data = pd.DataFrame(self.animals, columns=['Animal'])


    def get_animalInfo(self, root_folder):
        animalInfoFile = os.path.join(root_folder, 'animalInfo.csv')
        animalInfo = pd.read_csv(animalInfoFile)
        animals = animalInfo['AnimalID'].values
        geneBG = animalInfo['Genotype'].values
        if 'Sex' in animalInfo.keys():
            sex = animalInfo['Sex'].values
        else:
            sex = np.full(len(animals), np.nan)
        # convert animal to string
        animals = list(map(str, animals))

        output = {'animals': animals, 'geneBG': geneBG, 'sex': sex}
        return output

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
        # in 5 mins window
        runningAve_distance = np.full((self.minFrames - 1, self.nSubjects), np.nan)
        runningAve_velocity = np.full((self.minFrames - 1, self.nSubjects), np.nan)
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

            # running windows
            savefigFolder = os.path.join(self.analysisFolder, self.animals[idx])
            t = 5*60  # running windos of 5 mins
            obj.get_movement_running(t, savefigFolder)
            obj.get_angular_velocity_running(t, savefigFolder)

            runningAve_distance[0:len(obj.dist_running),idx]=obj.dist_running.flatten()
            runningAve_velocity[0:len(obj.dist_running), idx] = obj.vel_running.flatten()
        """ make plots"""
        """distance plot"""
        if 'KO' in np.unique(self.data['GeneBG']):
            mutLabel = 'KO'
        elif 'Mut' in np.unique(self.data['GeneBG']):
            mutLabel = 'Mut'

        # WTIdx = np.where(self.data['GeneBG'] == 'WT')[0]

        # plot the result without considering sex info
        # WTBoot = bootstrap(distanceMat[:, self.WTIdx], 1,
        #                        distanceMat[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(distanceMat[:, self.MutIdx], 1,
        #                         distanceMat[:, self.MutIdx].shape[0],500)
        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)

        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Distance travelled (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Distance traveled.tif', 'tif', savefigpath)
        # distPlot.save_plot('Distance traveled.svg', 'svg', savefigpath)
        #
        # """velocity plot"""
        # WTBoot = bootstrap(velocityDist[:, self.WTIdx], 1,
        #                        velocityDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(velocityDist[:, self.MutIdx], 1,
        #                         velocityDist[:, self.MutIdx].shape[0])
        # velPlot = StartPlots()
        # velPlot.ax.plot(velEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # velPlot.ax.fill_between(velEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # velPlot.ax.plot(velEdges, MutBoot['bootAve'], color=MutColor, label='KO')
        # velPlot.ax.fill_between(velEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # velPlot.ax.set_xlabel('Velocity (px/s)')
        # velPlot.ax.set_ylabel('Velocity distribution (%)')
        # velPlot.legend(['WT', 'KO'])
        # velPlot.save_plot('Velocity distribution.tif', 'tif', savefigpath)
        # velPlot.save_plot('Velocity distribution.svg', 'svg', savefigpath)
        #
        # """ plot angular velocity distribution"""
        # WTBoot = bootstrap(angularDist[:, self.WTIdx], 1,
        #                        angularDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(angularDist[:, self.MutIdx], 1,
        #                         angularDist[:, self.MutIdx].shape[0])
        #
        # """angular velocity plot"""
        # angPlot = StartPlots()
        # angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        # angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.set_xlabel('Angular velocity (radian/s)')
        # angPlot.ax.set_ylabel('Angular velocity distribution (%)')
        # angPlot.legend(['WT', 'Mut'])
        # angPlot.save_plot('Angular velocity distribution.tif', 'tif', savefigpath)
        # angPlot.save_plot('Angular velocity distribution.svg', 'svg', savefigpath)
        #
        # """plot head angular velocity distribution"""
        # WTBoot = bootstrap(headAngularDist[:, self.WTIdx], 1,
        #                        headAngularDist[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(headAngularDist[:, self.MutIdx], 1,
        #                         headAngularDist[:, self.MutIdx].shape[0])
        #
        # angPlot = StartPlots()
        # angPlot.ax.plot(angEdges, WTBoot['bootAve'], color=WTColor, label='WT')
        # angPlot.ax.fill_between(angEdges, WTBoot['bootLow'],
        #                             WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.plot(angEdges, MutBoot['bootAve'], color=MutColor, label='Mut')
        # angPlot.ax.fill_between(angEdges, MutBoot['bootLow'],
        #                             MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # angPlot.ax.set_xlabel('Angular velocity(head) (radian/s)')
        # angPlot.ax.set_ylabel('Angular velocity(head) distribution (%)')
        # angPlot.legend(['WT', 'Mut'])
        # angPlot.save_plot('Angular velocity(head) distribution.tif', 'tif', savefigpath)
        # angPlot.save_plot('Angular velocity(head distribution.svg', 'svg', savefigpath)
        #
        #
        # # distance and velocity in 5 mins running window
        # WTBoot = bootstrap(runningAve_distance[:, self.WTIdx], 1,
        #                        runningAve_distance[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(runningAve_distance[:, self.MutIdx], 1,
        #                         runningAve_distance[:, self.MutIdx].shape[0],500)
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Running average distance travelled in 5 mins (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Running average distance travelled in 5 mins.tif', 'tif', savefigpath)
        # distPlot.save_plot('Running average distance travelled in 5 mins.svg', 'svg', savefigpath)
        #
        # WTBoot = bootstrap(runningAve_velocity[:, self.WTIdx], 1,
        #                        runningAve_velocity[:, self.WTIdx].shape[0], 500)
        # MutBoot = bootstrap(runningAve_velocity[:, self.MutIdx], 1,
        #                         runningAve_velocity[:, self.MutIdx].shape[0],500)
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(self.plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(self.plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(self.plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(self.plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Running average velocity in 5 mins (px)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Running average distance travelled in 5 mins.tif', 'tif', savefigpath)
        # distPlot.save_plot('Running average distance travelled in 5 mins.svg', 'svg', savefigpath)
        #
        # plt.close('all')

        # plot result separating male and female
        # save distanceMat, runningAve_distance

        # convert to cm
        savedistPath = os.path.join(savefigpath, 'CumulativeDistance.csv')
        data = {}
        for idx,animal in enumerate(self.animals):
            data[animal] = distanceMat[:,idx]
        data['time'] = self.plotT
        data = pd.DataFrame(data)
        data.to_csv(savedistPath)

        savedistPath = os.path.join(savefigpath, 'runningAverageDistance.csv')
        data = {}
        for idx,animal in enumerate(self.animals):
            data[animal] = runningAve_distance[:,idx]
        data['time'] = self.plotT
        data = pd.DataFrame(data)
        data.to_csv(savedistPath)

        for ss in ['male','female', 'allsex']:
            # plot distance
            self.plot_movement_results(distanceMat,self.plotT,savefigpath,
                                       'Distance travelled', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)

            self.plot_movement_results(velocityDist,velEdges,savefigpath,
                                       'Velocity', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)
            self.plot_movement_results(angularDist,angEdges,savefigpath,
                                       'Angular velocity', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)
            self.plot_movement_results(runningAve_distance,self.plotT,savefigpath,
                                       'Distance running 5 mins', ss,
                                       ['WT', 'Mut'],WTColor, MutColor)


    def plot_movement_results(self, variableMat, plotT, savefigpath, label, group, leg,color1, color2):
        if group =='male':
        # if consider sex info
            WTIdx = list(set(self.WTIdx) & set(self.maleIdx))
            mutIdx = list(set(self.MutIdx) & set(self.maleIdx))
        elif group == 'female':
            WTIdx = list(set(self.WTIdx) & set(self.femaleIdx))
            mutIdx = list(set(self.MutIdx) & set(self.femaleIdx))
        elif group == 'allsex':
            WTIdx = self.WTIdx
            mutIdx = self.MutIdx
        WTBoot = bootstrap(variableMat[:, WTIdx], 1,
                               variableMat[:, WTIdx].shape[0], 200)
        MutBoot = bootstrap(variableMat[:, mutIdx], 1,
                                variableMat[:, mutIdx].shape[0],200)
        WTColor = color1
        MutColor = color2

        distPlot = StartPlots()
        distPlot.ax.plot(plotT, WTBoot['bootAve'], color=color1, label='WT')
        distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
                                     WTBoot['bootHigh'], color=color1, alpha=0.2, label='_nolegend_')
        distPlot.ax.plot(plotT, MutBoot['bootAve'], color=color2, label='KO')
        distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
                                     MutBoot['bootHigh'], color=color2, alpha=0.2, label='_nolegend_')
        distPlot.ax.set_xlabel('Time (s)')
        title = label + ' ' + group
        distPlot.ax.set_ylabel(title)
        distPlot.legend(leg)
        #distPlot.ax.set_ylim(0, np.nanmax(variableMat))
        # save the plot
        distPlot.save_plot(title+'.png', 'png', savefigpath)
        distPlot.save_plot(title+'.svg', 'svg', savefigpath)
        plt.close()

    def center_analysis(self, savefigpath):
        centerMat = np.full((self.minFrames, self.nSubjects), np.nan)
        runningAve_center = np.full((self.minFrames, self.nSubjects), np.nan)
        numCrossMat = np.full((self.minFrames, self.nSubjects), np.nan)
        plotT = np.arange(self.minFrames)/self.fps
        for idx, obj in enumerate(self.data['DLC_obj']):
            savefigFolder = os.path.join(self.analysisFolder, self.animals[idx])
            if not os.path.exists(savefigFolder):
                os.makedirs(savefigFolder)
            obj.moving_trace(savefigFolder)
            obj.get_time_in_center()
            t = 5*60
            obj.plot_distance_to_center(t, savefigFolder)
            centerMat[:,idx] = obj.cumu_time_center[0:self.minFrames]
            numCrossMat[:, idx] = obj.num_cross[0:self.minFrames]
            if idx==0:
                nbins = len(obj.dist_center_bins[1])
                centerDistMat = np.full((nbins-1, self.nSubjects), np.nan)
                centerDistMat30 = np.full((nbins - 1, self.nSubjects), np.nan)
            centerDistMat[:,idx] = obj.dist_center_bins[0]
            centerDistMat30[:,idx] = obj.dist_center_bins_30[0]

            runningAve_center[0: len(obj.dist_center_running), idx]=obj.dist_center_running.flatten()

        WTColor = (255 / 255, 189 / 255, 53 / 255)
        MutColor = (63 / 255, 167 / 255, 150 / 255)
        # save centerMat result

        # total time in the center
        totalCenter = centerMat[-1,:]
        totalCross = numCrossMat[-1,:]
        # violin plot
        custom_palette = {0: WTColor, 1: MutColor}
        ax=sns.violinplot(data=[totalCenter[self.WTIdx], totalCenter[self.MutIdx]],palette=custom_palette)
        ax.set_xticklabels(['WT', 'Mut'])
        ax.set_ylabel('Total time in the center')
        ax.set_xlabel('Group')
        ax.set_title('Total time in the center')
        plt.savefig(savefigpath + '\\violin_center_time.png', dpi=300)
        plt.savefig(savefigpath + '\\violin_center_time.svg', dpi=300)
        plt.close()

        ax=sns.violinplot(data=[totalCross[self.WTIdx], totalCross[self.MutIdx]],palette=custom_palette)
        ax.set_xticklabels(['WT', 'Mut'])
        ax.set_ylabel('Total cross time')
        ax.set_xlabel('Group')
        ax.set_title('Total cross time')
        plt.savefig(savefigpath + '/violin_cross_time.png', dpi=300)
        plt.savefig(savefigpath + '/violin_cross_time.svg', dpi=300)
        plt.close()

        data = {'animalID':self.animals,
                'timeinCenter': totalCenter,
                'crossTime': totalCross}
        data = pd.DataFrame(data)
        data.to_csv(savefigpath + '/timeinCenter.csv')

        # WTBoot = bootstrap(centerMat[:, self.WTIdx], 1,
        #                        centerMat[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(centerMat[:, self.MutIdx], 1,
        #                         centerMat[:, self.MutIdx].shape[0])
        # WTColor = (255 / 255, 189 / 255, 53 / 255)
        # MutColor = (63 / 255, 167 / 255, 150 / 255)
        #
        # binX = (obj.dist_center_bins[1][0:-1] + obj.dist_center_bins[1][1:]) / 2
        #
        # for ss in ['male','female','allsex']:
        #     # plot distance
        #     self.plot_movement_results(centerMat,plotT,savefigpath,
        #                                'Time spent in the center', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(centerDistMat,binX,savefigpath,
        #                                'Distribution of distance from center', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(centerDistMat30,binX,savefigpath,
        #                                'Distribution of distance from center 30 mins', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(runningAve_center,plotT,savefigpath,
        #                                'Time spent in the center in running 5 mins windows', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)
        #     self.plot_movement_results(numCrossMat,plotT,savefigpath,
        #                                'Num of crossings', ss,
        #                                ['WT', 'Mut'],WTColor, MutColor)

        # KS test
        # from scipy.stats import ks_2samp
        # WTMale = centerDistMat30[:,  list(set(self.WTIdx) & set(self.maleIdx))]
        # MutMale = centerDistMat30[:,  list(set(self.MutIdx) & set(self.maleIdx))]
        # WTFemale = centerDistMat30[:,  list(set(self.WTIdx) & set(self.femaleIdx))]
        # MutFemale = centerDistMat30[:,  list(set(self.MutIdx) & set(self.femaleIdx))]
        # # Assuming you have two arrays of data: data1 and data2
        # # Perform the KS test
        # statistic, p_value = ks_2samp(WTMaleBoot['bootAve'], MutMaleBoot['bootAve'])
        #
        # from scipy.stats import permutation_test
        # res = permutation_test((WTMale.T,MutMale.T), ks_2samp)
        #
        # # Print the test statistic and p-value
        # print("KS statistic:", statistic)
        # print("p-value:", p_value)
        # # two-way ANOVA for centerDistMat30
        # gene_anova_male = []
        # dist_anova_male = []
        # response_anova_male = []
        # subject_male = []
        # gene_anova_female = []
        # dist_anova_female = []
        # subject_female = []
        # response_anova_female = []
        #
        # for t in range(len(self.GeneBG)):
        #     for s in range(len(binX)):
        #         if self.Sex[t] == 'M':
        #             response_anova_male.append(centerDistMat30[s,t])
        #             gene_anova_male.append(self.GeneBG[t])
        #             dist_anova_male.append(binX[s])
        #             subject_male.append(self.animals[t])
        #         else:
        #             response_anova_female.append(centerDistMat30[s,t])
        #             gene_anova_female.append(self.GeneBG[t])
        #             dist_anova_female.append(binX[s])
        #             subject_female.append(self.animals[t])
        #
        # anova_data = pd.DataFrame({'gene': gene_anova_male,
        #                            'dist': dist_anova_male,
        #                            'response': response_anova_male,
        #                            'subject': subject_male
        #                            })
        # model = ols('response ~ gene + dist + gene:dist', anova_data).fit()
        # anova_table = sm.stats.anova_lm(model, typ=3)
        # # print ANOVA table
        # print(anova_table)
        #
        # # three way anova?
        # gene_anova = []
        # dist_anova = []
        # response_anova = []
        # subject = []
        # sex = []
        #
        #
        # for t in range(len(self.GeneBG)):
        #     for s in range(len(binX)):
        #         response_anova.append(centerDistMat30[s,t])
        #         gene_anova.append(self.GeneBG[t])
        #         dist_anova.append(binX[s])
        #         subject.append(self.animals[t])
        #         sex.append(self.Sex[t])
        #
        # anova_data = pd.DataFrame({'gene': gene_anova,
        #                            'dist': dist_anova,
        #                            'response': response_anova,
        #                            'sex': sex,
        #                            'subject': subject
        #                            })
        # model = ols('response ~ gene + dist + sex + gene:dist + gene:sex + dist:sex + dist:sex:gene', anova_data).fit()
        # anova_table = sm.stats.anova_lm(model, typ=3)
        # # print ANOVA table
        # print(anova_table)
        #
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Time spent in the center (s)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Time spent in the center.tif', 'tif', savefigpath)
        # distPlot.save_plot('Time spent in the center.svg', 'svg', savefigpath)
        #
        # # distribution of distance from center
        # WTBoot = bootstrap(centerDistMat[:, self.WTIdx], 1,
        #                        centerDistMat[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(centerDistMat[:, self.MutIdx], 1,
        #                         centerDistMat[:, self.MutIdx].shape[0])
        # binX = (obj.dist_center_bins[1][0:-1] + obj.dist_center_bins[1][1:])/2
        # distPlot = StartPlots()
        # distPlot.ax.plot(binX, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(binX, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(binX, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(binX, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Distance from center (px)')
        # distPlot.ax.set_ylabel('Number of frames')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Distribution of distance from center.tif', 'tif', savefigpath)
        # distPlot.save_plot('Distribution of distance from center.svg', 'svg', savefigpath)
        #
        # # plot average distance from center in running windows
        # WTBoot = bootstrap(runningAve_center[:, self.WTIdx], 1,
        #                        runningAve_center[:, self.WTIdx].shape[0])
        # MutBoot = bootstrap(runningAve_center[:, self.MutIdx], 1,
        #                         runningAve_center[:, self.MutIdx].shape[0])
        #
        # distPlot = StartPlots()
        # distPlot.ax.plot(plotT, WTBoot['bootAve'], color=WTColor, label='WT')
        # distPlot.ax.fill_between(plotT, WTBoot['bootLow'],
        #                              WTBoot['bootHigh'], color=WTColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.plot(plotT, MutBoot['bootAve'], color=MutColor, label='KO')
        # distPlot.ax.fill_between(plotT, MutBoot['bootLow'],
        #                              MutBoot['bootHigh'], color=MutColor, alpha=0.2, label='_nolegend_')
        # distPlot.ax.set_xlabel('Time (s)')
        # distPlot.ax.set_ylabel('Time spent in the center in running 5 mins windows (s)')
        # distPlot.legend(['WT', 'KO'])
        # # save the plot
        # distPlot.save_plot('Time spent in the center in running 5 mins windows.tif', 'tif', savefigpath)
        # distPlot.save_plot('Time spent in the center in running 5 mins windows.svg', 'svg', savefigpath)

class DLC_OF(DLCSummary):
    def __init__(self, root_folder, fps, groups,behavior):
        super().__init__(root_folder, fps, groups,behavior)  # Call the parent class's __init__ method
        DLC_results = []
        video = []
        animalID = []
        analysis = []
        GeneBGID = []
        sessionID = []
        sexID = []
        for aidx,aa in enumerate(self.animals):
            sessionPattern = r'_([0-9]{1,2})(?=DLC)'
            filePatternCSV = '*' + aa + '_OF_*.csv'
            filePatternVideo = '*' + aa + '*.mp4'
            csvfiles = glob.glob(f"{dataFolder}/{'DLC'}/{filePatternCSV}")
            if not csvfiles == []:
                for ff in range(len(csvfiles)):
                    DLC_results.append(csvfiles[ff])
                    video.append(glob.glob(f"{dataFolder}/{'Videos'}/{filePatternVideo}")[ff])
                    animalID.append(aa)

                    analysis.append(os.path.join(self.analysisFolder, aa))
                    sessionID.append(aa)
                    GeneBGID.append(self.GeneBG[aidx])
                    sexID.append(self.Sex[aidx])

        self.data = pd.DataFrame(animalID, columns=['Animal'])
        self.data['CSV'] = DLC_results
        self.data['Video'] = video

        self.data['AnalysisPath'] = analysis
        self.data['GeneBG'] = GeneBGID
        self.data['Sex'] = sexID
        self.nSubjects = len(self.animals)

        self.nSessions = len(self.data['Animal'])
        DLC_obj = []

        minFrames = 10 ** 8
        for s in range(self.nSessions):
            analysisPath = self.data['AnalysisPath'][s]
            filePath = self.data['CSV'][s]
            videoPath = self.data['Video'][s]
            dlc = DLCData(filePath, videoPath, analysisPath, fps)
            DLC_obj.append(dlc)
            if dlc.nFrames < minFrames:
                minFrames = dlc.nFrames

        self.minFrames = minFrames
        self.data['DLC_obj'] = DLC_obj
        self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSessions)
        self.WTIdx = animalIdx[self.data['GeneBG'] == groups[0]]
        self.MutIdx = animalIdx[self.data['GeneBG'] == groups[1]]
        # grouping the animals

        if self.Sex[0]==np.nan: # if no sex info
            nGroups = 1
        else:
            nGroups = 2


        if nGroups==2:
            self.maleIdx = np.where(self.data['Sex']=='M')[0]
            self.femaleIdx = np.where(self.data['Sex']=='F')[0]

class DLC_Rotarod(DLCSummary):

    def __init__(self, root_folder, fps, groups,behavior):
        super().__init__(root_folder, fps, groups,behavior)  # Call the parent class's __init__ method

        DLC_results = []
        stage = []
        Rod_speed = []
        timeStamp = []
        video= []
        animalID = []
        analysis = []
        GeneBGID = []
        sessionID = []
        sexID = []
        date = []
        timeOnRod = []
        fallbyTurning = []
        self.behaviorResults = os.path.join(root_folder, 'RR_results.csv')
        rr_results = pd.read_csv(self.behaviorResults)
        # %% load all files
        for aidx,aa in enumerate(self.animals):

            filePatternSpeed = '*ASD' + aa + '*.csv'
            filePatternDLC = '*ASD' + aa + '*.csv'
            filePatternVideo = '*ASD' + aa + '*.avi'
            filePatternTimestamp = '*ASD' + aa + '*.csv'

            speedCSV = glob.glob(f"{dataFolder}/{'Speed'}/{filePatternSpeed}")
            timeStampCSV = glob.glob(f"{dataFolder}/{'Videos'}/{filePatternTimestamp}")
            videoFiles = glob.glob(f"{dataFolder}/{'Videos'}/{filePatternVideo}")
            DLCFiles = glob.glob(f"{dataFolder}/{'DLC'}/{filePatternDLC}")
            num_files = len(videoFiles)

            if num_files>0:
                for ff in range(num_files):
                    # match the sessions
                    dateExpr = r'\d{6}_trial\d{1,2}'
                    matches = re.findall(dateExpr,videoFiles[ff][0:-23])
                    # in tempVideo['back'], find the string that has matches
                    video.append(videoFiles[ff])
                    DLC_ID = [ID for ID in range(len(DLCFiles)) if matches[0] in DLCFiles[ID]]
                    if len(DLC_ID)>0:
                        DLC_results.append(DLCFiles[DLC_ID[0]])
                    else:
                        DLC_results.append(None)
                    speed_ID = [ID for ID in range(len(speedCSV)) if matches[0] in speedCSV[ID]]
                    Rod_speed.append(speedCSV[speed_ID[0]])
                    timeStamp_ID = [ID for ID in range(len(timeStampCSV)) if matches[0] in timeStampCSV[ID]]
                    timeStamp.append(timeStampCSV[timeStamp_ID[0]])

                    animalID.append(aa)

                    #stage.append(matches[0])
                    analysis.append(os.path.join(self.analysisFolder, aa, matches[0]))
                    ses = re.search(r'\d{1,2}\s*$', matches[0])
                    sessionID.append(int(ses.group()))
                    date.append(matches[0][0:6])
                    GeneBGID.append(self.GeneBG[aidx])
                    sexID.append(self.Sex[aidx])

                    # find the animal and trial in rr_result
                    result = rr_results[(rr_results['animalID'].str.contains(aa)) & (rr_results['Trial'] == int(ses.group()))]
                    timeOnRod.append(int(result['Time']))
                    fallbyTurning.append(result['fall by turning'].astype(bool).values[0])

        self.data = pd.DataFrame(animalID, columns=['Animal'])
        self.data['DLC'] = DLC_results
        self.data['Video'] = video
        self.data['Rod_speed'] = Rod_speed
        self.data['AnalysisPath'] = analysis
        self.data['GeneBG'] = GeneBGID
        self.data['Sex'] = sexID
        self.data['Trial'] = sessionID
        self.data['Date'] = date
        self.data['Timestamp'] = timeStamp
        self.data['TimeOnRod'] = timeOnRod
        self.data['FallByTurning'] = fallbyTurning

        self.nSubjects = len(self.animals)
        sorted_df = self.data.sort_values(by=['Animal', 'Trial'])
        sorted_df = sorted_df.reset_index(drop=True)
        self.data=sorted_df
        self.nSessions = len(self.data['Animal'])

        #%% make a plot for rotarod behavior
        terminalVel = np.full((self.nSubjects, max(self.data['Trial'])), np.nan)
        for idx,aa in enumerate(self.animals):
            for tt in range(max(self.data['Trial'])):
                #if tt+1 <= 6:
                #    startRPM = 5
                #    endRPM = 40
                #else:
                startRPM = 5
                endRPM = 80
                totalTime = 300
                time = self.data['TimeOnRod'][np.logical_and(self.data['Animal'] == aa, self.data['Trial'] == tt+1)]
                if len(time)>0:
                    terminalVel[idx,tt]=((endRPM-startRPM)/totalTime)*time+startRPM
                else:
                    terminalVel[idx,tt]=np.nan

        #plt.figure()

        #savefigpath = os.path.join(self.sumFolder, 'terminalVelocity.svg')
        # convert time on rod to terminal velocity (RPM)

        #minFrames = min([len(dlc) for dlc in DLC_obj])

        #%% load the DLC data
        DLC_obj= []

        for s in range(self.nSessions):
            analysisPath = self.data['AnalysisPath'][s]

            filePath = self.data['DLC'][s]
            videoPath = self.data['Video'][s]
            rodPath = self.data['Rod_speed'][s]
            fps = self.data['Timestamp'][s]
            dlc = DLCData(filePath, videoPath, rodPath, analysisPath, fps)
            DLC_obj.append(dlc)

        self.data['DLC_obj'] = DLC_obj
        #self.plotT = np.arange(0, minFrames-1)/fps
        animalIdx = np.arange(self.nSessions)
        self.WTIdx = animalIdx[self.data['GeneBG'] == groups[0]]
        self.MutIdx = animalIdx[self.data['GeneBG'] == groups[1]]
        # grouping the animals

        if self.Sex[0]==np.nan: # if no sex info
            nGroups = 1
        else:
            nGroups = 2

        if nGroups==2:
            self.maleIdx = np.where(self.data['Sex']=='M')[0]
            self.femaleIdx = np.where(self.data['Sex']=='F')[0]

        self.startVoltage = [4.45, 40] # 5 rpm = 0.273 V
        self.endVoltage = [8.90, 80]
        self.rod_a = (self.endVoltage[1] - self.startVoltage[1]) / (self.endVoltage[0] - self.startVoltage[0])
        self.rod_b = self.endVoltage[1] - self.endVoltage[0] * self.rod_a

    def align(self):
        # preprocess data, align the video with rod speed.
        # we will need to manually label the number of frame when rod start to turn
        # load the csv files
        #stampCSV = os.path.join(self.dataFolder,'Videos', 'timeStamp.csv')
        #timeStamp = pd.read_csv(stampCSV)
        #self.data['rodData'] = [[] for x in range(self.nSessions)]

        # check if DLC data exists

        savefigpath = os.path.join(self.analysisFolder, 'StartPoint_rodSpeed')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)

        for ss in range(self.nSessions):
            savefigname = os.path.join(savefigpath, 'Start point for ' + str(self.data['Animal'][ss]) + ' trial' + str(self.data['Trial'][ss])+'.png')
            # check if fig exist
            DLC_obj = self.data['DLC_obj'][ss]
            savedataname = os.path.join(DLC_obj.analysis, 'smoothed_rodSpeed.csv')
            if not os.path.exists(savedataname):

                DLC_obj = self.data['DLC_obj'][ss]

                # Generate a sample signal with change points

                # Create a change point detection object using a specific algorithm (e.g., 'Pelt' or 'Binseg')
                signal = DLC_obj.data['rodSpeed']/100

                # downsample it

                algo = rpt.Pelt(model="l2").fit(signal)
                # Predict the change points
                predicted_bkps = algo.predict(pen=3)

                # Display results
                # change the signal to rod speed

                rodSpeed = signal*self.rod_a+self.rod_b
                if self.data['Trial'][ss] <=6:
                    max_jump = (40-5)/(60*5*4)
                else:
                    max_jump = (80-10)/(60*5*4)
                # Smooth the signal
                smoothed_signal = np.copy(rodSpeed)
                #running average first
                tempSpeed = rodSpeed
                #plt.figure()
                #plt.plot(rodSpeed)
                if rodSpeed[0] == 0 and rodSpeed[-1] == 0:
                    # steady state has been recorded
                    startRange = predicted_bkps[0]
                    endRange = predicted_bkps[-2]
                elif rodSpeed[0] == 0 and rodSpeed[-1] > 0:
                    startRange = predicted_bkps[0]
                    endRange = len(rodSpeed)
                elif  rodSpeed[0] > 0 and rodSpeed[-1] == 0:
                    # steady state hasn't been recorded
                    startRange = 0
                    endRange = predicted_bkps[-2]
                else:
                    startRange = 0
                    endRange = len(rodSpeed)

                for sss in range(0,60,10):
                    windowSize = 4+sss*1

                    for i in range(startRange, endRange):
                        if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
                        elif i <= startRange + windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
                        elif i >= endRange - windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])
                    tempSpeed = smoothed_signal
                    #plt.plot(tempSpeed)
                    # jump = smoothed_signal[i] - smoothed_signal[i - 1]
                    # if abs(jump) > max_jump:
                    #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
                        #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2

                algo = rpt.Pelt(model="l2").fit(smoothed_signal)
                # Predict the change points
                new_predicted_bkps = algo.predict(pen=1)

                # smooth one more round with the new predicted change point
                #plt.figure()
                #plt.plot(smoothed_signal)
                for sss in range(0,60,10):
                    windowSize = 4+sss*1
                    for i in range(startRange, endRange):
                        if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
                        elif i <= startRange + windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
                        elif i >= endRange - windowSize/2:
                            smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])

                    # jump = smoothed_signal[i] - smoothed_signal[i - 1]
                    # if abs(jump) > max_jump:
                    #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
                        #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2
                rodTime = DLC_obj.data['rodT']

                algo = rpt.Pelt(model="l2").fit(smoothed_signal)
                # Predict the change points
                new_predicted_bkps = algo.predict(pen=170)

                plt.figure()
                plt.plot(rodTime,rodSpeed)
                plt.plot(rodTime,smoothed_signal)
                plt.scatter(rodTime[new_predicted_bkps[0]], 0, s=200)
                plt.scatter(rodTime[new_predicted_bkps[1]], 0, s=200)
                plt.scatter(rodTime[new_predicted_bkps[-2]],0, s=200)
                plt.title('Start point for ' + str(self.data['Animal'][ss]) + ' trial' + str(self.data['Trial'][ss]))
                #plt.show()
                plt.savefig(savefigname)
                plt.close()


            # save the running speed and voltage
                # save the smoothed_signal somewhere
                saveData={}
                saveData['raw'] = signal
                saveData['smoothed'] = smoothed_signal
                saveData['time'] = rodTime
                if rodSpeed[0] ==0:
                    saveData['Start'] = np.zeros((len(rodTime)))+rodTime[new_predicted_bkps[0]]
                    saveData['Run'] = np.zeros((len(rodTime)))+rodTime[new_predicted_bkps[1]]
                elif rodSpeed[new_predicted_bkps[1]+10] == 0:
                    # stopped in the middle
                    saveData['Start'] = np.zeros((len(rodTime)))+rodTime[new_predicted_bkps[0]]
                    saveData['Run'] = np.zeros((len(rodTime)))+rodTime[new_predicted_bkps[3]]
                else:
                    saveData['Start'] = np.full((len(rodTime)),np.nan)
                    saveData['Run'] = np.zeros((len(rodTime)))+rodTime[new_predicted_bkps[0]]
                savedf= pd.DataFrame(saveData)
                savedf.to_csv(savedataname)

            else:
                saveData = pd.read_csv(savedataname)

            # double check for rod run (speed > 5 rpm for more than 10 seconds)
            above = saveData['smoothed'] > 5

            # Find the first index i such that all subsequent values stay > 5
            t0 = None
            for i in range(len(above)):
                if above[i] and np.all(above[i:i+500]):
                    t0 = saveData['time'][i]
                    break
            
            # if t0 and saveData['Run'][0] is close enough (1s), use t0
            #if t0 and abs(t0 - saveData['Run'][0]) < 1:
            saveData['Run'] = np.zeros((len(saveData['time']))) + t0

            self.data['DLC_obj'][ss].data['rodSpeed_smoothed'] = saveData['smoothed']
            self.data['DLC_obj'][ss].data['rodStart'] = saveData['Start']
            self.data['DLC_obj'][ss].data['rodRun'] = saveData['Run']
                # load it somewhere?

            #%% find the point when animal turn around
            # if DLC file exists:
            if self.data['DLC'][ss] is not None:
                tempData = self.data['DLC_obj'][ss].data
                ave_left_rod_back = np.array([np.mean(np.array(tempData['rod_left_back']['x'])[np.array(tempData['rod_left_back']['p'])>0.95]),
                                np.mean(np.array(tempData['rod_left_back']['y'])[np.array(tempData['rod_left_back']['p'])>0.95])])
                ave_right_rod_back = np.array([np.mean(np.array(tempData['rod_right_back']['x'])[np.array(tempData['rod_right_back']['p'])>0.95]),
                                np.mean(np.array(tempData['rod_right_back']['y'])[np.array(tempData['rod_right_back']['p'])>0.95])])
                ave_center_rod_back = (ave_left_rod_back+ave_right_rod_back)/2
                self.data['DLC_obj'][ss].data['left_rod_back'] = ave_left_rod_back
                self.data['DLC_obj'][ss].data['right_rod_back'] = ave_right_rod_back
                self.data['DLC_obj'][ss].data['center_rod_back'] = ave_center_rod_back

                ave_left_rod_front = np.array([np.mean(np.array(tempData['rod_left_front']['x'])[np.array(tempData['rod_left_front']['p'])>0.95]),
                                np.mean(np.array(tempData['rod_left_front']['y'])[np.array(tempData['rod_left_front']['p'])>0.95])])
                ave_right_rod_front = np.array([np.mean(np.array(tempData['rod_right_front']['x'])[np.array(tempData['rod_right_front']['p'])>0.95]),
                                np.mean(np.array(tempData['rod_right_front']['y'])[np.array(tempData['rod_right_front']['p'])>0.95])])
                ave_center_rod_front = (ave_left_rod_front+ave_right_rod_front)/2

                self.data['DLC_obj'][ss].data['left_rod_front'] = ave_left_rod_front
                self.data['DLC_obj'][ss].data['right_rod_front'] = ave_right_rod_front
                self.data['DLC_obj'][ss].data['center_rod_front'] = ave_center_rod_front

            # estimations on the left of 'right_rod_back' is in back area
            # on the right of 'right rod front' is in front area
            # plot a 1/0 mask to show each keypoints where they belong
                kp_list = tempData['bodyparts']
                viewMask = np.zeros((len(kp_list), len(tempData['rod_right_back']['x'])))
                # 1 for back 0 for front
                for idx,kp in enumerate(kp_list):
                # on the left of
                    viewMask[idx,:] = np.array(tempData[kp]['x']) < ave_right_rod_back[0]

                # plot frame with keypoints
                # frame_num = 6180
                # curr_frame = read_video(self.data['DLC_obj'][ss].videoPath, frame_num, ifgray=False)
                # plt.figure()
                # plt.imshow(curr_frame)
                # cmap = cm.get_cmap('viridis', len(kp_list))
                # for kp in kp_list:
                #     plt.scatter(tempData[kp]['x'][frame_num], tempData[kp]['y'][frame_num], c=cmap(kp_list.index(kp)), s=200,label = kp)
                #
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                # plot number of back/front keypoints that is actually in back/front view
                back_kp = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
                front_kp = ['spine 1', 'left ear', 'right ear', 'nose', 'left hand', 'right hand']
                viewNumber = np.zeros((3, len(tempData['rod_right_back']['x'])))
                for kp in kp_list:
                    if kp in back_kp:
                        viewNumber[0,:] = viewNumber[0,:] + viewMask[kp_list.index(kp),:]
                    elif kp in front_kp:
                        viewNumber[1,:] = viewNumber[1,:] + 1-viewMask[kp_list.index(kp),:]
                viewNumber[2,:] = (viewNumber[0,:] + viewNumber[1,:])/(len(back_kp)+len(front_kp))

                # plot number of back/front keypoints that is actually in back/front view
                # need to set some threshold to identify when the animal turn around
                # consistently smaller 50% for longer than 3 seconds?
                p_thresh = 0.6
                min_duration = 1 # seconds
                below_threshold = viewNumber[2,:] < p_thresh

                # Initialize start and end indices list
                segments = []
                start_idx = None

                for i in range(len(viewNumber[2,:])):
                    if below_threshold[i]:
                        # Start of a new segment
                        if start_idx is None:
                            start_idx = i
                    else:
                        # End of a segment
                        if start_idx is not None:
                            # Calculate segment duration
                            duration = tempData['time'][i - 1] - tempData['time'][start_idx]
                            if duration >= min_duration:
                                segments.append((start_idx, i - 1))
                            start_idx = None

                # Check if the last segment meets the minimum duration
                if start_idx is not None:
                    duration = tempData['time'][-1] - tempData['time'][start_idx]
                    if duration >= min_duration:
                        segments.append((start_idx, len(viewNumber[2,:]) - 1))

                saveFolder = os.path.join(self.analysisFolder, 'TurnAround')
                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder)
                savepath = os.path.join(saveFolder,
                               'turning period_' + self.data['Animal'][ss] + '_Trial ' + str(self.data['Trial'][ss]) + '.png')

                if not os.path.exists(savepath):

                    fig, ax = plt.subplots(3,1, figsize=(16, 8),
                                           gridspec_kw={'height_ratios': [3, 1, 1]})  # Adjust figure size for visibility
                    ax[0].imshow(viewMask, cmap='gray', aspect='auto', interpolation='none')
                    ax[0].set_yticks(ticks=np.arange(len(kp_list)), labels=kp_list)
                    #x_tick_interval = 50
                    #x_positions = np.where((tempData['time'] % x_tick_interval) < (x_tick_interval / len(tempData['time'])))[0]
                    #x_labels = [f"{int(tempData['time'][idx])}" for idx in x_positions]
                    #ax[0].set_xticks(ticks=x_positions, label                   ax[1].plot(tempData['time'], viewNumber[0,:],linewidth=1)
                    ax[1].plot(tempData['time'], viewNumber[1,:],linewidth=1)
                    ax[1].plot(tempData['time'], viewNumber[2,:],linewidth=1)
                    for tt in segments:
                        ax[1].axvspan(tempData['time'][tt[0]], tempData['time'][tt[1]], color='red', alpha=1)

                    ax[2].plot(tempData['rodT'],saveData['smoothed'])
                    # save the figure
                    fig.savefig(savepath, dpi=300, bbox_inches='tight')
                    plt.close()

                self.data['DLC_obj'][ss].data['turning_period'] = segments


    def stride_analysis(self,front_kp, back_kp):
        # stride analysis for rotarod behavior
        #strideMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)

        for idx, obj in enumerate(self.data['DLC_obj']):
            # check if DLC exist

            if self.data['DLC'][idx] is not None:
                obj.get_stride(front_kp, back_kp, self.data.iloc[idx])
        #
        # for idx, obj in enumerate(self.data['DLC_obj_back']):
        #     obj.get_stride('back')

    def stride_summary(self):
        # things to do:
        # 1. foot amplitude and frequency in the beginning (5-20 rpm)

        #%% average cross correlation
        """ calculate the average cross correlation with in speed interval """
        startSpeed = np.arange(10,80,10)
        nTrials = 12
        stride = {}
        amp_std = {}
        amp_std_running = {}
        stride_amp_running = {}
        stride_freq_running = {}
        plot_speed = [] # keep the longest window speed to plot
        time_step = 5 # 2 s window
        # load stride frequency data
        keys = ['left hand', 'right hand', 'left foot', 'right foot']
        for key in keys:
            amp_std[key] = np.full((self.nSubjects, nTrials),np.nan)
            amp_std_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]
            stride_amp_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]
            stride_freq_running[key] = [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)]

        amp_std['perf'] = np.full((self.nSubjects, nTrials),np.nan)
        genotype = self.GeneBG

        for idx, obj in enumerate(self.data['DLC_obj']):
            animal = self.data['Animal'][idx]
            trialIdx = self.data['Trial'][idx]-1
            animalIdx = self.animals.index(animal)

            if self.data['DLC'][idx] is not None:
                #%%  load the Stride_freq
                stridepickle = os.path.join(obj.analysis,'stride_freq.pickle')
                with open(stridepickle, 'rb') as handle:
                    stride = pickle.load(handle)
                rodSpeedCSV = os.path.join(obj.analysis, 'smoothed_rodSpeed.csv')
                rodSpeed = pd.read_csv(rodSpeedCSV)

                # isolate the time when animals turns around
                truncatedStride = copy.deepcopy(stride)
                bp_keys = ['left hand', 'right hand', 'left foot', 'right foot']
                for tInterval in obj.data['turning_period']:
                    tStart = max(stride['time'][0], obj.data['time'][tInterval[0]])
                    tEnd = min(stride['time'][len(stride['time'])-1],obj.data['time'][tInterval[1]])
                    nanMask = np.logical_and(stride['time']>=tStart, stride['time']<=tEnd)
                    for key in bp_keys:
                        truncatedStride[key][nanMask] = np.nan
                    
                    truncatedStride['stride frequency'][nanMask,:] = np.nan

                    # remove stride amplitude in the turning period
                    for sa in range(4):
                        nanMask_sa = np.logical_and(stride['stride time'][sa]>=tStart,
                                                    stride['stride time'][sa]<=tEnd)
                        truncatedStride['stride amplitude'][sa][nanMask_sa] = np.nan

                # calculate the average standard deviation of stride amplitude
                for kidx, key in enumerate(bp_keys):
                    amp_std[key][animalIdx, trialIdx] = np.nanstd(truncatedStride['stride amplitude'][kidx])
                
                if not self.data['FallByTurning'][np.logical_and(self.data['Animal']==animal,
                                                 self.data['Trial']==trialIdx+1)].any():
                    amp_std['perf'][animalIdx, trialIdx] = self.data['TimeOnRod'][np.logical_and(self.data['Animal']==animal,
                                                                                            self.data['Trial']==trialIdx+1)]
                
                # calculate running amplitude STD as a function of rod speed
                rod_time    = np.array(rodSpeed['time'])
                rod_speed   = np.array(rodSpeed['smoothed'])
                                    # --- Parameters ---
                window_size = 10.0    # 5 seconds window
                step_size   = 1    # sliding step 0.5 s

                # --- Determine start time: first time rod speed > 5 ---
                start_idx = np.where(rod_speed > 5)[0]
                if len(start_idx) == 0:
                    raise ValueError("Rod speed never exceeds 5")
                start_time = rod_time[start_idx[0]]

                # --- Generate sliding window time points ---
                below_zero_idx = np.where(rod_speed <= 0)[0]

                # Only consider "drops to zero" that happen *after* start_time
                below_zero_after_start = below_zero_idx[rod_time[below_zero_idx] > start_time]

                if len(below_zero_after_start) > 0:
                    # the first drop to zero after motion began
                    end_zero_time = rod_time[below_zero_after_start[0]]
                    end_time = end_zero_time - 2
                    # ensure it doesn’t go earlier than start_time
                    if end_time <= start_time:
                        end_time = rod_time[-1]
                else:
                    # if it never returns to zero
                    end_time = rod_time[-1]

                window_starts = np.arange(start_time, end_time - window_size + 0.01, step_size)
                
                for kidx, key in enumerate(bp_keys):
                    
                    stride_time = np.array(truncatedStride['stride time'][kidx])
                    stride_amp  = np.array(truncatedStride['stride amplitude'][kidx])


                    # --- Compute running amp, freq, and amp std for each window ---
                    running_std = []
                    running_amp = []
                    running_freq = []
                    window_centers = []

                    for t0 in window_starts:
                        t1 = t0 + window_size
                        mask = (stride_time >= t0) & (stride_time < t1)
                        if np.any(mask):
                            running_std.append(np.std(stride_amp[mask]))
                            running_amp.append(np.nanmean(stride_amp[mask]))
                            running_freq.append(np.sum(mask)/window_size)  # strides per second
                            window_centers.append(t0 + window_size/2)
                        else:
                            running_std.append(np.nan)  # no stride in window
                            running_amp.append(np.nan)
                            running_freq.append(np.nan)
                            window_centers.append(t0 + window_size/2)

                    # --- Convert to arrays for plotting ---
                    running_std = np.array(running_std)
                    running_amp = np.array(running_amp)
                    running_freq = np.array(running_freq)
                    window_centers = np.array(window_centers)
                    window_rod_speed = np.interp(window_centers, rod_time, rod_speed)
                    if len(window_rod_speed) > len(plot_speed):
                        plot_speed = window_rod_speed

                    amp_std_running[key][animalIdx][trialIdx]=running_std
                    stride_amp_running[key][animalIdx][trialIdx]=running_amp
                    stride_freq_running[key][animalIdx][trialIdx]=running_freq

                #%% load correlation 
                corrCSV = os.path.join(obj.analysis,'Stride correlation.csv')
                correlation = pd.read_csv(corrCSV)

                truncatedCorr = copy.deepcopy(correlation)
                corr_keys = correlation.keys().tolist()
                corr_keys.remove('time')
                for tInterval in obj.data['turning_period']:
                    tStart = max(stride['time'][0], obj.data['time'][tInterval[0]])
                    tEnd = min(stride['time'][len(stride['time'])-1],obj.data['time'][tInterval[1]])
                    nanMask = np.logical_and(correlation['time']>=tStart, correlation['time']<=tEnd)
                    for key in corr_keys:
                        truncatedCorr[key][nanMask] = np.nan

                # initialize variable in the beginning
                if idx == 0:
                    corr_summary = {}
                    for key in corr_keys:
                        corr_summary[key] =  [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)] 
                    corr_summary['rodSpeed'] =  [[[] for _ in range(nTrials)] for _ in range(self.nSubjects)] 

                # take only time within start_time and end_time
                time_mask = np.logical_and(truncatedCorr['time']>=start_time, truncatedCorr['time']<=end_time)
                for key in corr_keys:
                    corr_summary[key][animalIdx][trialIdx] = truncatedCorr[key][time_mask]
                # interpolate time to rod speed
                corr_speed = np.interp(correlation['time'][time_mask], rod_time, rod_speed)
                corr_summary['rodSpeed'][animalIdx][trialIdx] = corr_speed
                
              



        #%% convert list to matrix, padding with NaN
        running_SD_matrix = {}
        for key in bp_keys: # convert running_std to matrix
                # find maximum length
            max_len = max(len(trial) for subj in amp_std_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(amp_std_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_SD_matrix[key] = data_3d

        running_amp_matrix = {}
        for key in bp_keys: # convert running_std to matrix
            # find maximum length
            max_len = max(len(trial) for subj in stride_amp_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(stride_amp_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_amp_matrix[key] = data_3d

        running_freq_matrix = {}
        for key in bp_keys: # convert running_std to matrix
            # find maximum length
            max_len = max(len(trial) for subj in stride_freq_running[key] for trial in subj)

            # create padded matrix
            data_3d = np.full((self.nSubjects, nTrials, max_len), np.nan)

            for i, subj in enumerate(stride_freq_running[key]):
                for j, trial in enumerate(subj):
                    if len(trial) > 0:
                        data_3d[i, j, :len(trial)] = trial
            running_freq_matrix[key] = data_3d

        corr_summary_matrix = {}
        for key in corr_keys:
            # Determine a common rod speed grid for interpolation
            # You can use the min/max across all trials to define it
            all_speeds = []
            for subj, subj_corr in zip(corr_summary[key], corr_summary['rodSpeed']):
                for trial, trial_speed in zip(subj, subj_corr):
                    if len(trial) > 0:
                        all_speeds.append(trial_speed)
            all_speeds = np.concatenate(all_speeds)
            
            # Define a common rod speed vector (e.g., 0.1 step)
            rod_speed_grid = np.arange(np.nanmin(all_speeds), np.nanmax(all_speeds)+0.1, 0.1)

            nSubjects = len(corr_summary[key])
            nTrials   = max(len(subj) for subj in corr_summary[key])
            nSpeeds   = len(rod_speed_grid)

            # Initialize 3D matrix with NaNs
            data_3d = np.full((nSubjects, nTrials, nSpeeds), np.nan)

            for i, subj_corr in enumerate(corr_summary[key]):
                for j, trial_corr in enumerate(subj_corr):
                    trial_speed = corr_summary['rodSpeed'][i][j]  # corresponding rod speed for this trial
                    if len(trial_corr) > 0:
                        # Interpolate trial data onto common rod speed grid
                        data_interp = np.interp(
                            rod_speed_grid,         # new x (grid)
                            trial_speed,            # original x
                            trial_corr,             # original y
                            left=np.nan,            # pad out-of-bounds with NaN
                            right=np.nan
                        )
                        data_3d[i, j, :] = data_interp

            corr_summary_matrix[key] = (rod_speed_grid, data_3d)
        
 
        #%% plot performance - stride std correlation 
        for key in bp_keys:
            perf = np.array(amp_std['perf'])          # shape (15, 12)
            left_sd = np.array(amp_std[key])  # shape (15, 12)
            genotype = np.array(genotype)             # length 15, entries 'WT' or 'KO'

            # --- masks ---
            wt_mask = genotype == 'WT'
            ko_mask = genotype == 'KO'

            # --- flatten + remove NaN for correlation ---
            def clean_flatten(mask):
                x = perf[mask].flatten()
                y = left_sd[mask].flatten()
                valid = ~np.isnan(x) & ~np.isnan(y)
                return x[valid], y[valid]

            perf_wt, sd_wt = clean_flatten(wt_mask)
            perf_ko, sd_ko = clean_flatten(ko_mask)

            # --- Pearson correlation ---
            r_wt, p_wt = pearsonr(perf_wt, sd_wt)
            r_ko, p_ko = pearsonr(perf_ko, sd_ko)

            # --- plotting setup ---
            trials = np.arange(12)
            norm = Normalize(vmin=0, vmax=11)
            cmap_wt = plt.cm.Greys
            cmap_ko = plt.cm.Reds

            plt.figure(figsize=(7,6))

            # --- scatter points ---
            for i in range(len(genotype)):
                cmap = cmap_wt if genotype[i] == 'WT' else cmap_ko
                colors = cmap(norm(trials))
                for t in trials:
                    x, y = perf[i, t], left_sd[i, t]
                    if not np.isnan(x) and not np.isnan(y):
                        plt.scatter(y, x, color=colors[t], s=60, edgecolor='none')

            # --- correlation text ---
            plt.text(0.05, 0.95, f"WT: r={r_wt:.2f}, p={p_wt:.3g}", transform=plt.gca().transAxes,
                    color='black', fontsize=10, va='top')
            plt.text(0.05, 0.88, f"KO: r={r_ko:.2f}, p={p_ko:.3g}", transform=plt.gca().transAxes,
                    color='red', fontsize=10, va='top')

            # --- legend 1: genotype ---
            genotype_handles = [
                Patch(facecolor='black', label='WT'),
                Patch(facecolor='red', label='KO')
            ]
            legend1 = plt.legend(handles=genotype_handles, loc='upper right', frameon=False)

            # --- legend 2: trial gradient ---
            sm = ScalarMappable(norm=norm, cmap=cmap_ko)
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
            cbar.set_label('Trial #', rotation=270, labelpad=15)
            cbar.set_ticks([0, 3, 6, 9])
            cbar.set_ticklabels(['1', '4', '7', '10'])

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # --- labels ---
            plt.ylabel('Performance')
            plt.xlabel('Amplitude SD')
            plt.title('Performance vs ' + key + ' Amplitude SD')

            plt.gca().add_artist(legend1)
            plt.tight_layout()
            plt.show()

            # save fig in png and svg format
            savefigpath = os.path.join(self.sumFolder, 'Performance vs ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Performance vs ' + key + ' Amplitude SD.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot running std vs rod speed for different gnotype

        # mixed ANOVA 
        for key in bp_keys:

            # Example dimensions
            data_3d = running_SD_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax1.set_title('Stride variability ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Stride variability ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Amplitude SD.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + ' Amplitude SD.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average frequency and amplitude vs rod speed
        for key in bp_keys:

            # Example dimensions
            data_3d = running_amp_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride amplitude (mean ± STE)')
            ax1.set_title('Stride amplitude ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Average stride amplitude ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Average Amplitude.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + 'Average Amplitude.svg')
            plt.savefig(savefigpath, format='svg')

        for key in bp_keys:

            # Example dimensions
            data_3d = running_freq_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d.shape

            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d, axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'stride_SD': data_3d[i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['stride_SD', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("stride_SD ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues

            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel('Stride frequency (mean ± STE)')
            ax1.set_title('Stride frequency ' + key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['stride_SD']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title('Average stride frequency ' + key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + ' Average Frequency.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + 'Average Frequency.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average correlation vs rod speed for different genotype
        for key in corr_keys:

            # Example dimensions
            data_3d = corr_summary_matrix[key]  # shape: nSubjects x nTrials x nSpeeds
            nSubjects, nTrials, nSpeeds = data_3d[1].shape
            plot_speed = data_3d[0] # x axis
            # --- Step 1: Average over trials per subject ---
            mean_per_subject = np.nanmean(data_3d[1], axis=1)  # shape: nSubjects x nSpeeds

            rows = []
            for i in range(nSubjects):
                for t in range(nTrials):
                    for s in range(nSpeeds):
                        rows.append({
                            'subject': f'subj_{i}',
                            'genotype': genotype[i],
                            'trial': t+1,                 # trial as factor
                            'rod_speed': plot_speed[s],
                            'dependentVar': data_3d[1][i, t, s]
                        })

            df_long = pd.DataFrame(rows)
            df_long['genotype'] = pd.Categorical(df_long['genotype'], categories=['WT', 'KO'])

            # --- Step 0: Drop rows with NaN in relevant columns ---
            df_clean = df_long.dropna(subset=['dependentVar', 'genotype', 'rod_speed', 'trial'])
            # --- Step 3: Fit mixed-effects model ---
            # Random intercept per subject
            model = smf.mixedlm("dependentVar ~ genotype * rod_speed * trial", data=df_clean, groups=df_clean["subject"])
            result = model.fit()
            pvals = result.pvalues
            print(result.summary())
            # Safe lookups for each effect of interest
            def get_p(name):
                return pvals.get(name, np.nan)

            p_genotype = get_p('genotype[T.KO]')
            p_genotype_speed = get_p('genotype[T.KO]:rod_speed')
            p_trial = get_p('trial')
            p_genotype_trial = get_p('genotype[T.KO]:trial')


            # data_3d: shape (nSubjects, nTrials, nSpeeds)
            # genotype: list of 'WT' or 'KO', length nSubjects
            # plot_speed: array of speeds

            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            plt.figure(figsize=(15, 8))
            genotypes_unique = ['WT', 'KO']
            colors = {'WT': 'black', 'KO': 'red'}

            # 1️⃣ Left plot: rod_speed
            ax1 = plt.subplot(1, 2, 1)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('rod_speed')['dependentVar']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax1.plot(mean_vals.index, mean_vals.values, color=colors[g], label=g, linewidth=2)
                ax1.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax1.set_xlabel('Rod speed')
            ax1.set_ylabel(key+' (mean ± STE)')
            ax1.set_title(key + ' vs rod speed')
            ax1.legend()
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.text(0.95, 0.95,
                    f'Genotype p = {p_genotype:.3e}\nGenotype×Speed p = {p_genotype_speed:.3e}',
                    transform=ax1.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # 2️⃣ Right plot: trial
            ax2 = plt.subplot(1, 2, 2)
            for g in genotypes_unique:
                df_g = df_clean[df_clean['genotype'] == g]
                grouped = df_g.groupby('trial')['dependentVar']
                mean_vals = grouped.mean()
                ste_vals = grouped.std() / np.sqrt(grouped.count())
                ax2.plot(mean_vals.index, mean_vals.values, color=colors[g], linewidth=2)
                ax2.fill_between(mean_vals.index,
                                mean_vals - ste_vals,
                                mean_vals + ste_vals,
                                color=colors[g], alpha=0.3)

            ax2.set_xlabel('Trial')
            #ax2.set_ylabel('Stride amplitude SD (mean ± STE)')
            ax2.set_title(key + ' vs trial')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.text(0.95, 0.95,
                    f'Trial p = {p_trial:.3e}\nGenotype×Trial p = {p_genotype_trial:.3e}',
                    transform=ax2.transAxes, fontsize=20, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            plt.tight_layout()
            plt.show()

            savefigpath = os.path.join(self.sumFolder, 'Changes of ' + key + '.png')
            plt.savefig(savefigpath, dpi=300)
            savefigpath = os.path.join(self.sumFolder, 'Changes of  ' + key + '.svg')
            plt.savefig(savefigpath, format='svg')

        #%% plot average amplitude/frequency at 5-20 RPM within trial 1-3, 4-6, 7-9, and 10-12
        

    def process_for_moseq(self):
        # stride analysis for rotarod behavior
        savefilefolder = os.path.join(self.rootFolder,'DLCforMoseq')
        if not os.path.exists(savefilefolder):
            os.makedirs(savefilefolder)
        for idx, obj in enumerate(self.data['DLC_obj']):
            animal = self.data['Animal'][idx]
            trialIdx = self.data['Trial'][idx]-1
            animalIdx = self.animals.index(animal)

            if self.data['DLC'][idx] is not None:
                # load the Stride_freq
                DLCCSV = self.data['DLC'][idx]
                df = pd.read_csv(DLCCSV)
                timeStart = np.where(obj.data['time']>obj.data['rodRun'][0])[0][0]
                timeOnRod = self.data['TimeOnRod'][np.logical_and(self.data['Animal']==animal,
                         self.data['Trial']==trialIdx+1)]
                timeEnd = np.where(obj.data['time']<(obj.data['rodRun'][0]+timeOnRod)[idx])[0][-1]
                # isolate the time when animals turns around
                tempMask = np.logical_and(pd.to_numeric(df['scorer'][2:])>timeStart, pd.to_numeric(df['scorer'][2:])<=timeEnd)
                tempMask = [True, True]+ list(tempMask)
                df_filtered = df[tempMask]

                file_path = os.path.normpath(DLCCSV)

                # Extract the base filename without extension
                filename = os.path.splitext(os.path.basename(file_path))[0]
                savefilepath = os.path.join(savefilefolder, filename+'forMoseq.csv')
                df_filtered.to_csv(savefilepath, index=False)

if __name__ == "__main__":
    import matplotlib.font_manager as fm

    # Get the list of available font families
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # Check if Arial is available
    if not 'Arial' in available_fonts:
        plt.rcParams['font.family'] = 'Liberation Sans'

    root_dir = r'Z:\HongliWang\Rotarod\Cntnap_rotarod'
    #root_dir = r'Z:\HongliWang\openfield\Erin\openfield_MGRPR_cKO'
    dataFolder = os.path.join(root_dir,'Data')
    #animals = ['1795', '1804', '1805', '1825', '1827', '1829']
    #animals = ['M3', 'M4', 'M5', 'M6']
    # add animal identity
    #GeneBG = ['WT', 'Mut', 'Mut', 'WT']
    #GeneBG = ['WT', 'Mut', 'WT', 'Mut', 'Mut', 'WT']
    fps = 0

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
    savemotionpath = os.path.join(root_dir, 'Summary', 'DLC')
    groups = ['Ctrl', 'Exp']
    behavior = 'Rotarod'
    DLCSum = DLC_Rotarod(root_dir, fps, groups,behavior)
 # recorded at 4 Hz
    DLCSum.align()
    #DLCSum.get_result(self)

    back_keypoints = ['spine 3', 'tail 1', 'tail 2', 'tail 3', 'left foot', 'right foot']
    front_keypoints = ['nose', 'left ear', 'right ear', 'left hand', 'right hand']
    #DLCSum.stride_analysis(front_keypoints, back_keypoints)

    DLCSum.stride_summary()

    DLCSum.process_for_moseq()
    # basic motor-related analysis
    #
    DLCSum.center_analysis(savemotionpath)
    DLCSum.motion_analysis(savemotionpath)
    # analysis to do
    # moving trace ( require user input to mark the open field area)
    # time spend in the center (same)
    # tail movement in egocentric coordinates
    savefigpath = r'D:\openfield_MGRPR\old\Analysis\DLC\1595'
    tt=DLCSum.data['DLC_obj'][0].moving_trace(savefigpath)
    """ keypoint-moseq analysis"""
    # DLCSum.data['DLC_obj'][0].get_time_in_center()
    moseqResults = r'Z:\HongliWang\Rotarod\Rotarod_DLC\Data\Moseq\results.h5'
    MoseqData = Moseq(root_dir)

    session = MoseqData.sessions[0]
    fps = 30
    savefigpath = os.path.join(root_dir, 'Analysis', 'Moseq')

    MoseqData.get_syllables(DLCSum)
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