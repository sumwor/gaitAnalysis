# utility functions
import imageio
import numpy as np
from pyPlotHW import *
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import color


def video_dlc_moseq_label(DLC, Moseq, timeInterval, videoFilePath):
    # input
    # DLC: dlc object
    # Moseq: Moseq.data[session]['estimated_coordinates'] that corresponding to DLC object
    # generate pose estimation videos with velocity, head direction, angular velocity ploted simutaneously
    # timeInterval: time interval for the video
    # get number of frames based on timeInterval
    frameIdx = np.arange(DLC.nFrames)
    frames = frameIdx[np.logical_and(DLC.t >= timeInterval[0],
                                     DLC.t < timeInterval[-1])]
    plotTime = DLC.t[np.logical_and(DLC.t >= timeInterval[0],
                                     DLC.t < timeInterval[-1])]
    frameCount = 0

    labelPlot = StartPlots()

    # get color map
    colormap_name = 'viridis'  # Replace with the desired colormap name

    # Get the colormap
    cmap = cm.get_cmap(colormap_name)

    # Pick 8 colors from the colormap

    colors = [cmap(x) for x in np.linspace(0, 1, len(DLC.data['bodyparts']))]

    writer = animation.FFMpegWriter(fps=DLC.fps)
    with writer.saving(labelPlot.fig, videoFilePath, dpi=100):
        for f in tqdm(frames):
            # clear the last frame
            labelPlot.ax.clear()
            # gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,20])
            image = read_video(DLC.videoPath, f)
            # labelPlot.ax[3].subplot(gs[3])
            labelPlot.ax.imshow(image, cmap='gray')
            for idx, bb in enumerate(DLC.data['bodyparts']):
                labelPlot.ax.scatter(DLC.data[bb]['x'][f],
                                        DLC.data[bb]['y'][f], color=colors[idx])
                labelPlot.ax.scatter(Moseq[f,idx,0], Moseq[f,idx,1],
                                     color=colors[idx], marker='*')
            # labelPlot.legend(3, 0, self.data['bodyparts'])

            frameCount += 1
            writer.grab_frame()

def read_video(videoPath, frame, ifgray):
    # ifgray: if convert the image to grayscale
    vid = imageio.get_reader(videoPath)

        #for ii in tqdm(range(self.nFrames)):
    if ifgray:
        image = color.rgb2gray(vid.get_data(frame))
    else:
        image = vid.get_data(frame)
        #   [xdim, ydim] = image.shape
        #    if ii == 0:
        #        # get video dimensions
                #imageStack = np.zeros((xdim, ydim, self.nFrames))
        #        imageStack = []
        #    imageStack.append(image)
    return image

def frame_input(videoPath):
    """load the first frame of a video, get the coordinates of 4 user-defined points"""
    matplotlib.use('Qt5Agg')
    plt.ion()
    frame = read_video(videoPath, 0, ifgray=True)
    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.axis('off')

    ax.set_title('Please select 4 cornors, upper L -> upper R -> lower R -> lower L')
    points = []
    point_names = ['upper left', 'upper right', 'lower right', 'lower left']

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')

            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Add a button to confirm input after 4 clicks
    button_ax = plt.axes([0.8, 0.05, 0.1, 0.02])  # Button position [x, y, width, height]
    button = plt.Button(button_ax, 'Confirm')

    confirm_clicked = False

    def confirm_callback(event):
        global confirm_clicked
        if event.inaxes == button_ax:
            confirm_clicked = True
            print(confirm_clicked)
            plt.disconnect(cid)  # Disconnect the onclick event handler function
            plt.close()  # Close the plot window

    button.on_clicked(confirm_callback)

    plt.show(block=True)

    arena = {}
    for i, n in enumerate(point_names):
        arena[n] = points[i]

    return arena

if __name__=="__main__":
    video_path = r'Z:\HongliWang\openfield\cntnap\052323\M1615_OF_2305231606\M1615_OF_2305231606_DS_0.5.mp4'
    arena = frame_input(video_path)
    x=1