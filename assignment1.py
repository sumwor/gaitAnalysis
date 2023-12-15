# Hint: left click in front of a line to add a breakpoint
# or search "how to point breakpoint in pycharm" for more details
# then click "debug" button on top right corner (the button that looks like a bug)
# the program will run and stop at where your breakpoint is (if it encounters no errors before).
# you can
# 1) inspect variables in workspace
# 2) try command in "Python console (bottom tab)" and so on
# 3) run the command line by line to understand it better, as you can monitor the variables got updated


import numpy as np
import matplotlib.pyplot as plt
import csv
class DLCData:

    def __init__(self, filePath, videoPath, fps):
        self.filePath = filePath
        self.videoPath = videoPath
        self.nFrames = 0
        self.fps = fps
        # read data
        self.data = self.read_data()
        #self.video = self.read_video()

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

    def get_distance(self, data):
        # calculate the distance travelled

        # initiate a variable to save the distance
        distanceTravelled = np.zeros((len(data['time'])-1))
        # loop through every frame to calculate the distance
        for t in len([data['time']]-1):
            # distance = sqrt((y2-y1)^2 + (x2-x1)^2)
            x1 = data['spine 1']['x'][t]
            y1 =
            x2 = data['spine 2']['x'][t+1]
            y2 =
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distanceTravelled[t] = distance

        return distanceTravelled

    def plot_distance(self, distanceTravelled):
        # use matplotlib to plot the cumulative distance

        # with distanceTravelled (which is the distance travelled between two frames
        # calculate the cumulative distance traveled from time=0 to time = t for every time t
        cumulativeDist = []

        # code to calculate cumulative distance


        # use d
        plt.figure()
        plt.plot(cumulativeDist)
        plt.xlabel('Time')
        plt.ylabel('Distance travelled')
        plt.show()

if __name__ == "__main__":
    # steps:
    # 1. read the csv file
    #filePath = r'path/to/DLCFile.csv'  # change this into where you save the file
    filePath = r'Z:\HongliWang\openfield\cntnap_dlc\M1595_OF_2305241457_DS_0.5DLC_resnet50_openfieldJun9shuffle1_700000.csv'
    videoPath = []    # leave this blank for now, for the first assignement we don't need the video
    
    fps = 40 # the video is recorded in 40 frames per second
    
    # create a DLCData instance
    """ we did not cover the concept of class today
    for now, consider it as a module of our own, the DLCData module is designed to read a .csv file
    , then process it"""
    DLC = DLCData(filePath, videoPath, fps)

    # read the data
    # The function read_data in class DLCData is used to load the data in csv file, return
    # as a dictionary called "data"
    data = DLC.read_data()

    # step 2
    # calculate the cumulative moving distance
    # there are multiple bodyparts, for simplicity, we simply use the part "spine 1"
    # fine x and y coordinates in of "spine 1" in "data" dictionary
    # the distance moved between two consecutive frame scan be calculated with:
    # distance = sqrt((y2-y1)^2 + (x2-x1)^2)
    # the cumulative distance at frame t = sum(distance(frame[0]-frame[t])
    distanceTravelled = DLC.get_distance(data)

    # step 3: plot the distance travelled as a function of time
    DLC.plot_distance(distanceTravelled)
    