# save videos in a different speed

import cv2
import os
import glob

def change_video_speed(input_dir, output_dir, speed):
    # accepted video extensions
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")
    
    # collect all video files
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print("No video files found in the folder.")
        return

    print(f"Found {len(video_files)} videos.")

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open: {video_path}")
            continue

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # output filename
        base = os.path.basename(video_path)
        name, ext = os.path.splitext(base)
        output_path = os.path.join(output_dir, f"{name}_{speed}{ext}")

        # New FPS (slower)
        new_fps = fps * speed
        print(f"Processing {base}: {fps:.2f} → {new_fps:.2f} FPS")

        out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

        # Write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        print(f"Saved: {output_path}")

# Example usage
input_dir = 'Z:\HongliWang\Rotarod\Cntnap_rotarod\moseq\my_models-3\grid_movies'
output_dir = 'Z:\HongliWang\Rotarod\Cntnap_rotarod\moseq\my_models-3\grid_movies_slow'
change_video_speed(input_dir, output_dir, speed=0.2)
