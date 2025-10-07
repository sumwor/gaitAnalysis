import sys
import os
import cv2
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QListWidget,
    QMessageBox,QListWidgetItem, QLineEdit,QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QIcon, QPixmap, QFont
import pyqtgraph as pg
from matplotlib import cm

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import re
import numpy as np

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Body Part Overlay")
        #self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.resize(800, 600)

        # Initialize main vertical layout
        self.layout = QVBoxLayout()

        # Button to select root directory
        self.folder_button = QPushButton("Select Root Directory")
        self.folder_button.clicked.connect(self.select_root_directory)
        self.layout.addWidget(self.folder_button)

        # Dropdown for selecting a video file
        self.video_dropdown = QComboBox()
        self.video_dropdown.currentIndexChanged.connect(self.handle_video_selection)
        self.layout.addWidget(self.video_dropdown)

        # Horizontal layout for video display and legend list
        video_layout = QHBoxLayout()

        # Video display label
        self.video_label = QLabel(self)
        #self.video_label.setFixedSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label)  # Add video label to the left

        # Legend list for body parts and likelihood
        legend_layout = QVBoxLayout()
        self.legend_list = HoverableListWidget()
        legend_layout.addWidget(self.legend_list)  # Add legend list to the right

        # Add foot stride plot widget below the legend
        self.foot_stride_widget = pg.PlotWidget()
        self.foot_stride_widget.setBackground('w')
        self.foot_stride_widget.addLegend()
        self.foot_stride_widget.setYRange(-50, 200, padding=0)
        legend_layout.addWidget(self.foot_stride_widget)
        #self.init_foot_stride_plot()

        # Add syllable plot widget below the foot stride
        self.syllable_canvas = FigureCanvas(Figure(figsize=(1, 2)))
        self.syllable_canvas.setFixedHeight(70)
        legend_layout.addWidget(self.syllable_canvas)
        self.init_syllable_plot()

        legend_layout.setStretch(0, 3)  # Legend list
        legend_layout.setStretch(1, 1)  # Foot stride canvas
        legend_layout.setStretch(2, 1)  # Syllable canvas

        video_layout.addLayout(legend_layout)

        # Initialize foot stride plot


        self.layout.addLayout(video_layout)

        # Play/pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.go_to_prev_frame)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_to_next_frame)

        self.save_button = QPushButton("Save Video")
        self.save_button.clicked.connect(self.save_video)
        self.layout.addWidget(self.save_button)

        # Slider for video navigation
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.seek_frame)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame number")

        self.go_button = QPushButton("Go to Frame")
        self.go_button.clicked.connect(self.go_to_frame)

        # Horizontal layout for play button and slider
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.frame_input)
        control_layout.addWidget(self.go_button)

        # Add control layout to main layout
        self.layout.addLayout(control_layout)

        # Set the main layout
        self.setLayout(self.layout)

        # Timer for video playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_video)

        # Video-related attributes
        self.root_dir = None
        self.video_file = None
        self.cap = None
        self.is_playing = False
        self.frame_index = 0
        self.total_frames = 0
        self.frame_rate = 0
        self.body_parts_data = None
        self.desired_scale = 0.5 # scale the video to fit the screen
        screen_geometry = self.screen().availableGeometry()
        self.max_width, self.max_height = screen_geometry.width(), screen_geometry.height()


    def select_root_directory(self):
        """Opens a folder dialog and populates the dropdown with video files in the selected directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Root Directory")
        if folder:
            self.root_dir = folder
            video_folder = os.path.join(self.root_dir, "Data", "Videos")
            if os.path.exists(video_folder) and os.listdir(video_folder):
                self.populate_video_files()
            else:
                QMessageBox.critical(self, "Error", "No videos found in Data/Videos inside the selected directory.")
                return
        else:
            QMessageBox.warning(self, "Warning", "No directory selected.")

    def populate_video_files(self):
        """Populates the dropdown with video files in the 'Data/Videos' folder inside the root directory."""
        try:
            self.video_dropdown.clear()
            video_folder = os.path.join(self.root_dir, "Data", "Videos")
            video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
            if video_files:
                self.video_dropdown.addItems(video_files)
            else:
                QMessageBox.critical(self, "Error", "No video files found in the folder.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video files: {e}")

    def handle_video_selection(self):
        """Handles video selection and loads the video and corresponding body part data."""
        video_filename = self.video_dropdown.currentText()
        if video_filename:
            video_path = os.path.join(self.root_dir, "Data", "Videos", video_filename)
            csv_path = self.find_corresponding_csv(video_filename)
            if csv_path:
                self.load_body_part_data(csv_path)

            match = re.search(r'^(.*)(?=\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})', video_filename).group(1)
            animal = re.search(r'ASD(\d+)', match).group(1)
            trial = re.search('trial(\d{1,2})(?!20\d{2})', match).group(1)
            datematch = re.search(r'(\d{4})-(\d{2})-(\d{2})T\d{2}_\d{2}_\d{2}', video_filename)
            trialFolder = datematch.group(1)[-2:]+datematch.group(2)+datematch.group(3)+'_trial'+trial
            stride_path = os.path.join(self.root_dir, 'Analysis', animal, trialFolder,'stride_freq.csv')
            rodSpeed_path = os.path.join(self.root_dir, 'Analysis', animal, trialFolder,'smoothed_rodSpeed.csv')

            self.load_stride_data(stride_path)

            self.load_rodSpeed_data(rodSpeed_path)

            # load video timestamp
            timeStampCSV = os.path.join(self.root_dir, "Data", "Videos",
                                        'ASD'+animal+'_'+datematch.group(1)[-2:]+datematch.group(2)+datematch.group(3)+'_trial'+trial+'_timeStamp.csv')
            time_raw = pd.read_csv(timeStampCSV, header=None)
            self.timeStamp = np.array(time_raw[0] - time_raw[0][0]) / 1000

            self.load_video(video_path)

            self.update_frame()

    def find_corresponding_csv(self, video_filename):
        """Finds the corresponding CSV file based on the video filename."""
        base_name = video_filename.split('.')[0]
        dlc_folder = os.path.join(self.root_dir, "Data", "DLC")
        for f in os.listdir(dlc_folder):
            if f.startswith(base_name) and f.endswith('.csv'):
                return os.path.join(dlc_folder, f)
        QMessageBox.warning(self, "Warning", "No matching CSV file found for the selected video.")
        return None

    def load_body_part_data(self, csv_path):
        """Loads body part data from the CSV file."""
        #try:
        df = pd.read_csv(csv_path, header=[1, 2])
        self.body_parts = [bp for bp, _ in df.columns[1::3]]
        self.body_parts_data = df.iloc[:,1:].values.reshape(-1, len(self.body_parts), 3)
        #except Exception as e:
        #    QMessageBox.critical(self, "Error", f"Failed to load body part data: {e}")

    def load_stride_data(self, stride_path):
        if os.path.exists(stride_path):
            df = pd.read_csv(stride_path)
            self.stride_freq = df
        else:
            self.stride_freq = None


    def load_rodSpeed_data(self, rodSpeed_path):
        if os.path.exists(rodSpeed_path):
            df = pd.read_csv(rodSpeed_path)
            self.rodSpeed = df
        else:
            self.rodSpeed = None


    def load_video(self, video_path):
        """Load the video file and setup video stream and slider."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video.")
            return

        # Get total frames and frame rate
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        # Set slider range
        self.slider.setMaximum(self.total_frames - 1)
        self.frame_index = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        self.update_frame()

        # Adjust the main window size to the content
        #self.adjust_window_size()

    # def adjust_window_size(self):
    #     """Resize the video display to fit within the screen dimensions."""
    #     # Get current frame dimensions
    #     video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    #     # Calculate scale factor to fit the video within screen limits
    #     scale_factor = min(self.max_width / video_width, self.max_height / video_height, 1)
    #     scaled_width = int(video_width * scale_factor)
    #     scaled_height = int(video_height * scale_factor)
    #
    #     # Resize the video display
    #     self.video_label.setFixedSize(QSize(scaled_width, scaled_height))
    #     self.resize(scaled_width, scaled_height)

    def update_frame(self):
        """Load and display the current frame with body part overlays in the QLabel."""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return

            # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
             # Replace with your desired size
            qimg = qimg.scaled(w*self.desired_scale, h*self.desired_scale,  Qt.AspectRatioMode.KeepAspectRatio)

            curr_time = self.timeStamp[self.frame_index]
            rodIndex = np.argmin(np.abs(self.rodSpeed['time'] - curr_time))
            if self.rodSpeed is not None:
                curr_speed = int(np.round(self.rodSpeed['smoothed'][rodIndex]))
            else:
                curr_speed = None

            # Overlay body parts, frame number, and rod speed on the QImage
            if self.body_parts_data is not None and self.frame_index < len(self.body_parts_data):
                painter = QPainter(qimg)
                # set pen color without boundary
                #painter.setPen(Qt.PenStyle.NoPen)
                colormap = cm.get_cmap('viridis', len(self.body_parts))

                for idx, body_part in enumerate(self.body_parts):
                    x, y, p = self.body_parts_data[self.frame_index, idx]
                    x=x*self.desired_scale
                    y=y*self.desired_scale
                    rgba = colormap(idx)
                    color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
                                   int(p * 255))  # Apply transparency based on likelihood
                    border_color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
                                   255)
                    #if p >= 0.5:  # Display only if likelihood is reasonably high
                    painter.setBrush(color)
                    painter.setPen(border_color)
                    painter.drawEllipse(int(x), int(y), 25*self.desired_scale, 25*self.desired_scale)

                # display frame index
                painter.setPen(QColor(0, 0, 0))  # Black color for text
                painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))  # Font size and style

                # Draw frame index on the top right of the frame
                frame_text = f"Frame: {self.frame_index}"
                text_rect = painter.boundingRect(0, 0, qimg.width()/2, qimg.height(), Qt.AlignmentFlag.AlignRight,
                                                 frame_text)
                painter.drawText(text_rect.right() - 100, 30, frame_text)  # Adjust position as needed
                if curr_speed is not None:
                    speed_text = f"Speed: {curr_speed} RPM"
                text_rect = painter.boundingRect(0, 0, qimg.width()/2, qimg.height(), Qt.AlignmentFlag.AlignRight,
                                                 frame_text)
                painter.drawText(text_rect.right() + 100, 30, speed_text)
                painter.end()


                # Update legend with body part names and likelihood
                self.legend_list.clear()
                for idx, body_part in enumerate(self.body_parts):
                    # set color
                    x, y, p = self.body_parts_data[self.frame_index, idx]
                    x=x*self.desired_scale
                    y=y*self.desired_scale
                    rgba = colormap(idx)
                    color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255)

                    # Create a small color icon for each body part
                    pixmap = QPixmap(20, 20)
                    pixmap.fill(Qt.GlobalColor.transparent)  # Transparent background
                    icon_painter = QPainter(pixmap)
                    icon_painter.setBrush(color)
                    icon_painter.setPen(Qt.PenStyle.NoPen)
                    icon_painter.drawEllipse(0, 0, 20*self.desired_scale, 20*self.desired_scale)
                    icon_painter.end()


                    item = QListWidgetItem(f"{body_part}: p={p:.2f}")
                    item.setIcon(QIcon(pixmap))  # Set the icon for the list item
                    self.legend_list.addItem(item)

            # Display the frame
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

            # Update slider position
            self.slider.setValue(self.frame_index)
            if self.is_playing:
                self.frame_index += 1

            # %% update stride, rod speed data

            self.update_foot_stride_plot(curr_time)

            # If video writer is initialized, write the current frame to the output video
            if hasattr(self, 'video_writer'):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)



            #self.frame_index += 1


    def seek_frame(self):
        """Seek to a specific frame when the slider is adjusted."""
        self.frame_index = self.slider.value()
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            self.update_frame()

    def go_to_prev_frame(self):
        """go to the previous frame"""
        self.frame_index = self.frame_index-1
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            self.update_frame()

    def go_to_next_frame(self):
        """go to the previous frame"""
        self.frame_index = self.frame_index+1
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            self.update_frame()

    def go_to_frame(self):
        """ go to a specific frame when the frame is specified in an input box"""
        self.frame_index = int(self.frame_input.text())
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            self.update_frame()

    def toggle_playback(self):
        """Toggle play/pause state."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
            self.frame_index -= 2
        else:
            # Set the timer interval based on the frame rate
            if self.frame_rate > 0:
                interval = int(1000 / self.frame_rate)
                self.timer.start(interval)
                self.play_button.setText("Pause")
        self.is_playing = not self.is_playing

    def play_video(self):
        """Play video frame by frame, updating the slider and display."""
        if self.frame_index < self.total_frames:
            self.update_frame()
        else:
            # Stop playback at the end of the video
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False

    def highlight_body_part(self, hovered_index):
        """Highlight the corresponding dot for the hovered body part in the frame."""

        if self.body_parts_data is not None and self.frame_index < len(self.body_parts_data):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            #self.frame_index = self.frame_index - 1
            ret, frame = self.cap.read()


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Use your frame data
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            qimg = qimg.scaled(w*self.desired_scale, h*self.desired_scale,  Qt.AspectRatioMode.KeepAspectRatio)

            # Overlay body parts on the QImage
            painter = QPainter(qimg)
            painter.setPen(Qt.PenStyle.NoPen)
            colormap = cm.get_cmap('viridis', len(self.body_parts))

            # set color
            #x, y, p = self.body_parts_data[self.frame_index, hovered_index]
            #rgba = colormap(hovered_index)
            #color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(p * 255))

            for idx, body_part in enumerate(self.body_parts):
                x, y, p = self.body_parts_data[self.frame_index, idx]
                x=x*self.desired_scale
                y=y*self.desired_scale
                rgba = colormap(idx)
                color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
                               int(p * 255))  # Apply transparency based on likelihood
                border_color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
                                      255)
                # if p >= 0.5:  # Display only if likelihood is reasonably high
                painter.setBrush(color)
                painter.setPen(border_color)
                painter.drawEllipse(int(x), int(y), 25*self.desired_scale, 25*self.desired_scale)

                # If the hovered index matches the current body part, highlight it
                if idx == hovered_index:
                    highlight_color = QColor(255, 255, 0)  # Yellow for highlighting
                    painter.setBrush(highlight_color)
                    painter.drawEllipse(int(x), int(y), 30*self.desired_scale, 30*self.desired_scale)  # Slightly larger circle to highlight

            # Draw frame index on the top right of the frame
            curr_time = self.timeStamp[self.frame_index]
            rodIndex = np.argmin(np.abs(self.rodSpeed['time'] - curr_time))
            if self.rodSpeed is not None:
                curr_speed = int(np.round(self.rodSpeed['smoothed'][rodIndex]))
            else:
                curr_speed = None

            painter.setPen(QColor(0, 0, 0))  # Black color for text
            painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))  # Font size and style

            frame_text = f"Frame: {self.frame_index}"
            text_rect = painter.boundingRect(0, 0, qimg.width() / 2, qimg.height(), Qt.AlignmentFlag.AlignRight,
                                             frame_text)
            painter.drawText(text_rect.right() - 100, 30, frame_text)
            if curr_speed is not None:
                speed_text = f"Speed: {curr_speed} RPM"
            text_rect = painter.boundingRect(0, 0, qimg.width() / 2, qimg.height(), Qt.AlignmentFlag.AlignRight,
                                             frame_text)
            painter.drawText(text_rect.right() + 100, 30, speed_text)

            painter.end()

            # Display the updated frame
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

    def init_foot_stride_plot(self):
        """Initialize the foot stride plot."""
        # check if the data exist

        ax = self.foot_stride_canvas.figure.add_subplot(111)
        ax.set_title("Foot Stride")
        ax.set_ylabel("Foot\n Stride (px)")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove ticks and labels
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        #ax.plot([], [])  # Placeholder for actual data
        self.foot_stride_canvas.draw()

    def update_foot_stride_plot(self, curr_time):
        self.foot_stride_widget.clear()
        mid_index = np.argmin(np.abs(self.stride_freq['time'] - curr_time))
        start_index = np.argmin(np.abs(self.stride_freq['time'] - curr_time+2))
        end_index = np.argmin(np.abs(self.stride_freq['time'] - curr_time-2))

        timeData = np.array(self.stride_freq['time'][start_index:end_index])
        leftFoot = np.array(self.stride_freq['left foot'][start_index:end_index])
        rightFoot = np.array(self.stride_freq['right foot'][start_index:end_index])

        self.foot_stride_widget.setXRange(self.stride_freq['time'][start_index],
                                           self.stride_freq['time'][end_index],padding=0)

        self.line1 = self.foot_stride_widget.plot(timeData,leftFoot, pen=pg.mkPen(color='r', width=2),
                                                  name = 'Left foot')
        self.line2 = self.foot_stride_widget.plot(timeData,rightFoot, pen=pg.mkPen(color='b', width=2),
                                                  name = 'Right foot')
        self.line3 = self.foot_stride_widget.plot([curr_time, curr_time], [-50,200], pen=pg.mkPen(color='black', width=2))
    def init_syllable_plot(self):
        """Initialize the foot stride plot."""
        ax = self.syllable_canvas.figure.add_subplot(111)
        ax.set_ylabel("Syllable")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Remove ticks and labels
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        #ax.plot([], [])  # Placeholder for actual data
        self.syllable_canvas.draw()

    def save_video(self):
        if not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "No video loaded.")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4);;AVI Files (*.avi)")
        if output_path:
            # Create VideoWriter object with the same frame size and frame rate as the input video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi files
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.frame_rate,
                                                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            QMessageBox.information(self, "Info", f"Saving video to {output_path}")

            # Now, we'll start processing the video and save the frames
            self.is_playing = True
            self.timer.start(int(1000 / (self.frame_rate*10)))  # Start playback
            #self.timer.start()
            self.play_button.setText("Pause")

class HoverableListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking

    # def enterEvent(self, event):
    #         # Get the item under the mouse cursor
    #     item = self.itemAt(event.pos())
    #     if item:
    #             # Get the index of the hovered item
    #         hovered_index = self.row(item)
    #         self.parent().highlight_body_part(hovered_index)

    #def leaveEvent(self, event):
    #    super().leaveEvent(event)
    #    self.parent().clear_highlight()

    def mouseMoveEvent(self, event):
        # Get the item under the mouse cursor
        item = self.itemAt(event.pos())
        if item:
            # Get the index of the hovered item
            hovered_index = self.row(item)
            self.parent().highlight_body_part(hovered_index)
        super().mouseMoveEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())
