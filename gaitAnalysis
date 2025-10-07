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
from scipy.signal import spectrogram
#import fitz
#from PIL import Image

from tqdm import tqdm
from pyPlotHW import *
from utils import *
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from tqdm import tqdm
from utility_HW import bootstrap, butter_lowpass_filter
import h5py
import statsmodels.api as sm
from statsmodels.formula.api import ols
import io
import subprocess as sp
import multiprocessing
import concurrent.futures
import functools
import seaborn as sns

# todo:
# 1. check the open field paper for related plots

class DLCData:

    def __init__(self, filePath, videoPath, analysisPath, fps):
        self.filePath = filePath
        self.videoPath = videoPath
        self.nFrames = 0
        self.fps = fps
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
        self.t = []
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

    def get_stride(self):
        savedatapath = os.path.join(self.analysis, 'stride_freq.csv')
        if not os.path.exists(savedatapath):
            self.stride = np.zeros((self.nFrames, 2))  # distance between spine and left/right foot
            for ff in range(self.nFrames):
                self.stride[ff,0] = np.sqrt((self.data['spine']['x'][ff]-self.data['left foot']['x'][ff])**2 +
                                            (self.data['spine']['y'][ff]-self.data['left foot']['y'][ff])**2)
                self.stride[ff,1] = np.sqrt((self.data['spine']['x'][ff]-self.data['right foot']['x'][ff])**2+
                                            (self.data['spine']['y'][ff]-self.data['right foot']['y'][ff])**2)
            self.filtered_stride = self.stride
            self.filtered_stride[:,0] = butter_lowpass_filter(self.stride[:,0], 10,self.fps,order=5)
            self.filtered_stride[:,1] = butter_lowpass_filter(self.stride[:, 1], 10, self.fps, order=5)

            f, t_spec, Sxx_left = spectrogram(self.filtered_stride[:,0], self.fps)
            f, t_spec, Sxx_right = spectrogram(self.filtered_stride[:, 1], self.fps)
            # Plot spectrogram

            # generate some plots
            fig = plt.figure(figsize=(20, 16))
            plot_time = 10
            # Subplot 1 (First row, spanning two columns)
            ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
            ax1.plot(self.t, self.filtered_stride[:,0])
            ax1.plot(self.t, self.filtered_stride[:,1])
            ax1.set_title('Distance between left/right foot and spine')

            # Subplot 2 (Second row, first column)
            ax2 = plt.subplot2grid((4, 2), (1, 0))
            ax2.plot(self.t[0:plot_time *self.fps], self.filtered_stride[0:plot_time*self.fps,0])
            ax2.plot(self.t[0:plot_time *self.fps], self.filtered_stride[0:plot_time *self.fps,1])
            ax2.set_title('Distance between foot and spine in the first'+str(plot_time) + ' seconds')

            # Subplot 3 (Second row, second column)
            ax3 = plt.subplot2grid((4, 2), (1, 1))
            ax3.plot(self.t[-plot_time  * self.fps:], self.filtered_stride[-plot_time  * self.fps:, 0])
            ax3.plot(self.t[-plot_time  * self.fps:], self.filtered_stride[-plot_time  * self.fps:, 1])
            ax3.set_title('Distance between foot and spine in the last' +str(plot_time) + ' seconds')

            # Subplot 4 (Third row, first column)
            ax4 = plt.subplot2grid((4, 2), (2, 0))
            ax4.pcolormesh(t_spec, f[0:40], 10 * np.log10(Sxx_left[0:40,:]), shading='auto')
            #ax4.colorbar(label='Power/Frequency (dB/Hz)')
            ax4.set_ylabel('Frequency (Hz)')
            ax4.set_title('Spectrogram of left stride')


            # Subplot 5 (Third row, second column)
            ax5 = plt.subplot2grid((4, 2), (2, 1))
            ax5.pcolormesh(t_spec, f[0:40], 10 * np.log10(Sxx_right[0:40,:]), shading='auto')

            ax5.set_title('Spectrogram of left stride')
            ax5.set_title('Spectrogram of right stride')

            timeStep = 5 # in second
            nWindows = int(np.floor(self.t[-1]/timeStep))
            corrCoeff = np.zeros((nWindows))
            tAxis = np.zeros((nWindows))
            # calculate the stride length
            for nn in range(nWindows):
                tStart = timeStep * nn * self.fps
                tEnd = timeStep * (nn+1) * self.fps-1
                corrCoeff[nn] = np.corrcoef(self.filtered_stride[tStart:tEnd,0], self.filtered_stride[tStart:tEnd,1])[0,1]
                tAxis[nn] = timeStep*nn
            corrCoeff_running = np.zeros((len(self.t)))
            for idx,t in enumerate(self.t):
                tMask = np.logical_and(self.t>t, self.t<t+timeStep)
                corrCoeff_running[idx] = np.corrcoef(self.filtered_stride[tMask,0], self.filtered_stride[tMask,1])[0,1]

            ax6 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
            ax6.plot(self.t, corrCoeff_running)
            ax6.plot([0,self.t[-1]], [0,0], 'k--')
            ax6.set_ylim([-1, 1])
            ax6.set_title('Correlation coefficient between two legs')
            ax6.set_xlabel('Time (s)')
            plt.tight_layout()  # Adjust subplot parameters to give specified padding
            plt.savefig(os.path.join(self.analysis,'Stide frequency analysis.png'), dpi=300)  # Save as PNG fil
            #plt.show()
            plt.close()

            # cross correlation in 10 second window

            # save data in csv
            data = {'stride_left': self.stride[:,0],
                    'stride_right': self.stride[:,1],
                    'correlation': corrCoeff_running,
                    'time': self.t}
            dataDF = pd.DataFrame(data)
            dataDF.to_csv(savedatapath)
            # with open(savedatapath, 'wb') as f:
            #     pickle.dump(self.stride, self. f)
            # f.close()
        else:
            print("Analysis already done")

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
        DLC_results = []
        video = []
        animalID = []
        analysis = []
        GeneBGID = []
        sessionID = []
        sexID = []
        for aidx,aa in enumerate(self.animals):
            if self.behavior == "openfield":
                filePatternCSV = '*' + aa + '_OF_*.csv'
            elif self.behavior == "Rotarod":
                filePatternCSV = '*' + aa + '*_Rotarod*.csv'
            filePatternVideo = '*' + aa + '*.mp4'
            sessionPattern = r'_([0-9]{1,2})(?=DLC)'
            csvfiles = glob.glob(f"{dataFolder}/{'DLC'}/{filePatternCSV}")
            if not csvfiles == []:
                for ff in range(len(csvfiles)):
                    DLC_results.append(csvfiles[ff])
                    video.append(glob.glob(f"{dataFolder}/{'Videos'}/{filePatternVideo}")[ff])
                    animalID.append(aa)
                    if self.behavior == "Rotarod":
                        matches = re.findall(sessionPattern, csvfiles[ff])
                        analysis.append(os.path.join(self.analysisFolder, aa,matches[0]))
                        sessionID.append(matches[0])
                    else:
                        analysis.append(os.path.join(self.analysisFolder, aa))
                        sessionID.append(aa)
                    GeneBGID.append(self.GeneBG[aidx])
                    sexID.append(self.Sex[aidx])
        DLC_results = []
        video = []
        animalID = []
        analysis = []
        GeneBGID = []
        sessionID = []
        sexID = []
        for aidx,aa in enumerate(self.animals):
            if self.behavior == "openfield":
                filePatternCSV = '*' + aa + '_OF_*.csv'
            elif self.behavior == "Rotarod":
                filePatternCSV = '*' + aa + '*_Rotarod*.csv'
            filePatternVideo = '*' + aa + '*.mp4'
            sessionPattern = r'_([0-9]{1,2})(?=DLC)'
            csvfiles = glob.glob(f"{dataFolder}/{'DLC'}/{filePatternCSV}")
            if not csvfiles == []:
                for ff in range(len(csvfiles)):
                    DLC_results.append(csvfiles[ff])
                    video.append(glob.glob(f"{dataFolder}/{'Videos'}/{filePatternVideo}")[ff])
                    animalID.append(aa)
                    if self.behavior == "Rotarod":
                        matches = re.findall(sessionPattern, csvfiles[ff])
                        analysis.append(os.path.join(self.analysisFolder, aa,matches[0]))
                        sessionID.append(matches[0])
                    else:
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

        self.nSessions = len(DLC_results)
        DLC_obj = []
        minFrames = 10 ** 8
        for s in range(self.nSessions):
            filePath = self.data['CSV'][s]
            videoPath = self.data['Video'][s]
            analysisPath = self.data['AnalysisPath'][s]
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

    def stride_analysis(self,savefigpath):
        # stride analysis for rotarod behavior
        #strideMat = np.full((self.minFrames - 1, self.nSubjects), np.nan)

        for idx, obj in enumerate(self.data['DLC_obj']):
            obj.get_stride()

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

if __name__ == "__main__":
    root_dir = r'Z:\HongliWang\openfield\Ahtesha\DLC_project'
    #root_dir = r'Z:\HongliWang\openfield\Erin\openfield_MGRPR_cKO'
    dataFolder = os.path.join(root_dir,'Data')
    #animals = ['1795', '1804', '1805', '1825', '1827', '1829']
    #animals = ['M3', 'M4', 'M5', 'M6']
    # add animal identity
    #GeneBG = ['WT', 'Mut', 'Mut', 'WT']
    #GeneBG = ['WT', 'Mut', 'WT', 'Mut', 'Mut', 'WT']
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
    savemotionpath = os.path.join(root_dir, 'Summary', 'DLC')
    groups = ['Ctrl', 'Exp']
    behavior = 'openfield'
    DLCSum = DLCSummary(root_dir, fps, groups,behavior)

    # basic motor-related analysis
    #
    DLCSum.center_analysis(savemotionpath)
    DLCSum.stride_analysis(savemotionpath)

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