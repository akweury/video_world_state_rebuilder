
import os
import cv2 as cv
from pathlib import Path
import shutil
from dataclasses import dataclass




# -------------- Path Settings --------------
root = Path(__file__).parents[0]
print(f"\n##### Root path: {root}\n")

exp_config_path = root / "experiments"


# -------------- System Settings --------------
BDD100K_PATH = root / "data" / "bdd100k" /"videos" / "train"
BDD100K_FRAMES_PATH = root / "data" / "bdd100k" /"frames"
BDD100K_FRAME_RATE = 5


# -------------- Inputs --------------
def get_step_01_input(args):

    data_num = args.data_num

    if args.dataset == "bdd100k":
        video_dir = BDD100K_PATH
        frame_dir = BDD100K_FRAMES_PATH
        frame_rate = BDD100K_FRAME_RATE
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    os.makedirs(frame_dir, exist_ok=True)

    all_video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    all_frame_paths = [os.path.join(frame_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]

    if data_num != "all":
        data_num = int(data_num)
        all_video_paths = all_video_paths[:data_num]
        all_frame_paths = all_frame_paths[:data_num]

    input_data = {
        "video_path": all_video_paths,
        "frame_path": all_frame_paths,
        "frame_rate": frame_rate
    }
    return input_data



