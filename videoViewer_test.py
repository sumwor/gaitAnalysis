import sys
import os
import cv2
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QListWidget,
    QMessageBox,QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QIcon, QPixmap, QFont
from matplotlib import cm


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Body Part Overlay")
        self.setGeometry(100, 100, 1000, 600)

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
        video_layout.addWidget(self.video_label)  # Add video label to the left

        # Legend list for body parts and likelihood
        self.legend_list = HoverableListWidget(self)
        video_layout.addWidget(self.legend_list)  # Add legend list to the right

        # Add video and legend layout to main layout
        self.layout.addLayout(video_layout)

        # Play/pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)

        # Slider for video navigation
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.seek_frame)

        # Horizontal layout for play button and slider
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.slider)

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
            self.load_video(video_path)

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
        # try:
        df = pd.read_csv(csv_path, header=[1, 2])
        self.body_parts = [bp for bp, _ in df.columns[1::3]]
        self.body_parts_data = df.iloc[3:, 1:].values.reshape(-1, len(self.body_parts), 3)
        # except Exception as e:
        #    QMessageBox.critical(self, "Error", f"Failed to load body part data: {e}")

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

            # Overlay body parts on the QImage
            if self.body_parts_data is not None and self.frame_index < len(self.body_parts_data):
                painter = QPainter(qimg)
                # set pen color without boundary
                painter.setPen(Qt.PenStyle.NoPen)
                colormap = cm.get_cmap('viridis', len(self.body_parts))

                for idx, body_part in enumerate(self.body_parts):
                    x, y, p = self.body_parts_data[self.frame_index, idx]
                    rgba = colormap(idx)
                    color = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
                                   int(p * 255))  # Apply transparency based on p
                    # if p >= 0.5:  # Display only if likelihood is reasonably high
                    painter.setBrush(color)
                    painter.drawEllipse(int(x), int(y), 25, 25)

                # display frame index
                painter.setPen(QColor(0, 0, 0))  # Black color for text
                painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))  # Font size and style

                # Draw frame index on the top right of the frame
                frame_text = f"Frame: {self.frame_index}"
                text_rect = painter.boundingRect(0, 0, qimg.width() / 2, qimg.height(), Qt.AlignmentFlag.AlignRight,
                                                 frame_text)
                painter.drawText(text_rect.right() - 100, 30, frame_text)  # Adjust position as needed

                painter.end()

            # Display the frame
            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

            # Update slider position
            self.slider.setValue(self.frame_index)

    def seek_frame(self):
        """Seek to a specific frame when the slider is adjusted."""
        self.frame_index = self.slider.value()
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            self.update_frame()

    def toggle_playback(self):
        """Toggle play/pause state."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            # Set the timer interval based on the frame rate
            if self.frame_rate:
                self.timer.start(1000 // self.frame_rate)
                self.play_button.setText("Pause")
        self.is_playing = not self.is_playing

    def play_video(self):
        """Play video by updating the frame index at the set frame rate."""
        if self.frame_index < self.total_frames - 1:
            self.frame_index += 1
            self.update_frame()

    def highlight_body_part(self, body_part_idx):
        """Highlight the body part when hovered over in the legend."""
        # Use a different method here for hover, without triggering frame change.
        self.update_frame()  # Just highlight without changing the frame index.

    class HoverableListWidget(QListWidget):
        def __init__(self, parent=None):
            super().__init__(parent)

        def enterEvent(self, event):
            super().enterEvent(event)
            self.parent().highlight_body_part(event)

        def leaveEvent(self, event):
            super().leaveEvent(event)
            self.parent().clear_highlight()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())