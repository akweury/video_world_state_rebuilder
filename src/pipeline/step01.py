
import os
from tqdm import tqdm

from src import config




def _video_to_frames(video_path, output_dir):
    """
    Convert a video into frames and save them as images in the output directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
    """
    import cv2
    import os

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()




def main(input_data):
    print("\n------- Step 01 -------\n")
    all_video_paths = input_data["video_path"]
    all_frame_paths = input_data["frame_path"]
    for v_path,f_path in tqdm(zip(all_video_paths, all_frame_paths), desc="video to frames"):
        _video_to_frames(v_path, f_path)

    print("\n--------- Step 01 Done ---------------\n")
    return 