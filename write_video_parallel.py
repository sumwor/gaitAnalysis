import imageio
import multiprocessing

# Function to process each frame and write to video
def process_frame(frame_path, output_path, frame_number):
    # Perform processing on the frame (e.g., apply filters, transformations, etc.)
    frame = imageio.imread(frame_path)

    # Write the frame to the video
    with output_lock:
        output_frames[frame_number] = frame

    print(f"Processed frame {frame_number}")

# Path to the input video
input_video_path = r'D:\openfield_cntnap\Data\videos\M1595_OF_2305241457_DS_0.5.mp4'

# Path to the output video
output_video_path = r'D:\openfield_cntnap\test.mp4'

# Number of worker processes
num_workers = multiprocessing.cpu_count()  # Use all available CPU cores

# Read the input video
input_video = imageio.get_reader(input_video_path)

# Get video properties
fps = input_video.get_meta_data()['fps']
frame_shape = input_video.get_data(0).shape

# Create the output video writer
output_video = imageio.get_writer(output_video_path, fps=fps)

# Create a lock to synchronize writing to the output video
output_lock = multiprocessing.Lock()

# Create a pool of worker processes
pool = multiprocessing.Pool(num_workers)

frame_number = 0
for frame in input_video:
    # Process each frame in parallel
    pool.apply_async(process_frame, args=(frame, frame_number))
    frame_number += 1

# Wait for all worker processes to finish
pool.close()
pool.join()

# Close the input video and release resources
input_video.close()

# Close the output video
output_video.close()

print("Video processing complete")