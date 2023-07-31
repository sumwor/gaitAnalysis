import imageio
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import glob

from gaitAnalysis import DLCData, Moseq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import numpy as np

class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Player")

        self.video_path = ""
        self.video = None
        self.total_frames = 0
        self.current_frame_index = 0

        # Create the main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(padx=0,pady=0)

        # Frame input box
        self.frame_input = tk.Entry(self.main_frame)
        self.frame_input.grid(row=0, column=0, padx=10)

        # Frame display button
        self.display_button = tk.Button(self.main_frame, text="Display Frame", command=self.display_input_frame)
        self.display_button.grid(row=0, column=1, padx=10)

        # Video selection button
        self.select_button = tk.Button(self.main_frame, text="Select Video", command=self.select_video)
        self.select_button.grid(row=0, column=2, padx=10)

        # Create a frame to hold the canvas and info textbox
        self.frame_container = tk.Frame(self.master)
        self.frame_container.pack(pady=0)

        # Create a canvas to display the video frame
        self.canvas_width = 960
        self.canvas_height = 720
        self.canvas = tk.Canvas(self.frame_container, width=self.canvas_width, height=self.canvas_height)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Create a frame_container to hold the info textbox, and syllable display (previous, current, next)
        #self.info_frame = tk.Frame(self.frame_container)
        #self.info_frame.pack(padx=(5,5))

        # Create a textbox to display additional information
        self.canvas_prev = tk.Canvas(self.frame_container, relief=tk.GROOVE,width=30, height=15)
        self.canvas_prev.grid(row=0, column=1, padx=10, pady=5)
        self.canvas_curr = tk.Canvas(self.frame_container, width=30, height=15)
        self.canvas_curr.grid(row=1, column=1, padx=10, pady=5)
        self.canvas_next = tk.Canvas(self.frame_container, width=30, height=15)
        self.canvas_next.grid(row=2, column=1, padx=10, pady=5)


    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:

            # load dlc and moseq data as well
            path = os.path.normpath(self.video_path)
            folders = path.split(os.sep)

            self.animal = folders[-1][0:5]
            dlcPath = os.path.join(folders[0], folders[1], folders[2], 'DLC')
            dlcPattern = self.animal+'*.csv'
            self.dlc_path = glob.glob(f"{dlcPath}/{dlcPattern}")[0]

            moseqPath = os.path.join(folders[0], folders[1], folders[2], 'Moseq')
            moseqPattern = 'results.h5'
            self.moseq_path = glob.glob(f"{moseqPath}/{moseqPattern}")[0]
            syllablePath = os.path.join(folders[0], folders[1], folders[2], 'Moseq', 'trajectory_plots')
            syllablePattern = 'Syllable*.gif'
            self.syllable_path = glob.glob(f"{syllablePath}/{syllablePattern}")

            self.load_video()

    def load_video(self):
        self.video = imageio.get_reader(self.video_path, "ffmpeg")
        self.total_frames = len(self.video)

        # load dlc and moseq results simutaneously
        self.dlcObj = DLCData(self.dlc_path,self.video_path,40)
        self.moseqObj = Moseq(self.moseq_path)

        # get the moseq data
        for key in self.moseqObj.data.keys():
            if self.animal in key:
                self.moseqData = self.moseqObj.data[key]

    def display_frame(self, frame_index):
        # Read the frame from the video
        frame = self.video.get_data(frame_index)

        # Calculate the original frame width and height
        frame_height, frame_width, _ = frame.shape

        # Calculate the width and height to maintain the aspect ratio
        if frame_width > frame_height:
            width = self.canvas_width
            height = int(frame_height * self.canvas_width / frame_width)
        else:
            height = self.canvas_height
            width = int(frame_width * self.canvas_height / frame_height)

        # Resize the frame to fit the calculated dimensions
        frame_resized = Image.fromarray(frame).resize((width, height))

        # Create a PhotoImage object from the resized frame
        photo = ImageTk.PhotoImage(image=frame_resized)

        # Clear the canvas and display the frame
        self.canvas.delete("all")
        self.canvas.create_image((self.canvas_width - width) // 2, (self.canvas_height - height) // 2,
                                 anchor="nw", image=photo)
        self.canvas.image = photo  # Store the reference to prevent image garbage collection

        # Display additional information in the info textbox
        #self.info_text.delete("1.0", tk.END)
        #self.info_text.insert(tk.END, f"Frame Index: {frame_index}\n")
        #self.info_text.insert(tk.END, f"Video Path: {self.video_path}\n")
        #if self.dlcObj:
        #    self.info_text.insert(tk.END, f"DLC Loaded\n")
        #if self.moseqData:
        #    self.info_text.insert(tk.END, f"Moseq Loaded\n")
        # Add more info as needed
        # Plot something on the frame
        self.plot_keypoint(frame, frame_index)

    def plot_keypoint(self, frame, frame_index):
        # Create a sample plot
        fig, ax = plt.subplots()
        ax.imshow(frame)
        bodyparts = self.dlcObj.data['bodyparts']

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

        colormap_name = 'viridis'  # Replace with the desired colormap name
        # Get the colormap
        cmap = cm.get_cmap(colormap_name)
        colors = [cmap(x) for x in np.linspace(0, 1, len(bodyparts))]

        for idx, bp in enumerate(bodyparts):
            ax.scatter(self.dlcObj.data[bp]['x'][frame_index],
                                 self.dlcObj.data[bp]['y'][frame_index], s=10, color=colors[idx])
            for skel in skeleton:
                if bp in skel:
                    ax.plot([self.dlcObj.data[skel[0]]['x'][frame_index],
                             self.dlcObj.data[skel[1]]['x'][frame_index]],
                            [self.dlcObj.data[skel[0]]['y'][frame_index],
                             self.dlcObj.data[skel[1]]['y'][frame_index]])
        # plot the skeleton as well
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Hide the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        # Convert the plot to an image
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_image = ImageTk.PhotoImage(Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb()))

        # Display the plot image on the canvas
        self.canvas.create_image((self.canvas_width - plot_image.width()) // 2, (self.canvas_height - plot_image.height()) // 2,
                                 anchor="nw", image=plot_image)
        self.canvas.image = plot_image

    def display_input_frame(self):
        try:
            frame_index = int(self.frame_input.get())
            if 0 <= frame_index < self.total_frames:
                self.display_frame(frame_index)
            else:
                print("Invalid frame number")
        except ValueError:
            print("Invalid frame number")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()