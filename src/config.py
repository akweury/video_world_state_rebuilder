
import os
import cv2 as cv
from pathlib import Path
import shutil
from dataclasses import dataclass

# -------------- Settings --------------
ALLOW_MODEL_DOWNLOAD = True

# -------------- Path Settings --------------
root = Path(__file__).parents[0]
print(f"\n##### Root path: {root}\n")

exp_config_path = root / "experiments"


# -------------- Inputs --------------
def step_0_setup(args):
    # get the dataset phyiscal path
    # ml-pulsar
    if args.machine == "ml-pulsar":
        print("\n##### Running on ml-pulsar #####\n")
        args.device = "cuda:0"
        args.dataset_path = Path('/home/sha/mnt/remote/dgx-g/storage-01/CauVid_Data/driving_mini')
        args.output_dir = root / 'output'/ 'bdd100k' / args.exp

    # dgx
    elif args.machine == "dgx":
        print("\n##### Running on DGX #####\n")
        raise ValueError("DGX is not supported yet. Please use ml-pulsar or jst.")
    
    # macbook pro
    elif args.machine == "macbook-pro":
        print("\n##### Running on MacBook Pro #####\n")
        args.device = "cpu"
        raise ValueError("MacBook Pro is not supported yet. Please use ml-pulsar or jst.")

    # JST
    elif args.machine == "jst":
        print("\n##### Running on JST #####\n")
        args.device = "cuda:0"
        raise ValueError("JST is not supported yet. Please use ml-pulsar or macbook-pro.")
    else:
        raise ValueError(f"Unsupported machine: {args.machine}")


    # test the dataset path
    if os.path.exists(args.dataset_path):
        print(f"\n##### Dataset path exists: {args.dataset_path} #####\n")
        # print the number of videos in the dataset path
        video_path = os.path.join(args.dataset_path, "videos")
        frames_path = os.path.join(args.dataset_path, "frames")
        depth_maps_path = os.path.join(args.dataset_path, "depth_maps")
        if os.path.exists(video_path):
            print(f"\n##### Number of videos in the dataset path: {len(os.listdir(video_path))} #####\n")
        else:
            raise ValueError(f"Video path does not exist: {video_path}")
        if os.path.exists(frames_path):
            print(f"\n##### Number of frames in the dataset path: {len(os.listdir(frames_path))} #####\n")
        else:
            raise ValueError(f"Frames path does not exist: {frames_path}")
        if os.path.exists(depth_maps_path):
            print(f"\n##### Number of depth maps in the dataset path: {len(os.listdir(depth_maps_path))} #####\n")
        else:
            raise ValueError(f"Depth maps path does not exist: {depth_maps_path}")
    else:
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")

    return args

def get_step_01_input(args):

    data_num = args.data_num

    if args.dataset == "bdd100k":
        video_dir = args.dataset_path / "videos"
        frame_dir = args.dataset_path / "frames"
        depth_dir = args.dataset_path / "depth_maps"
        flow_dir = args.dataset_path / "flows"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    os.makedirs(frame_dir, exist_ok=True)

    all_video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    all_frame_paths = [os.path.join(frame_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
    all_depth_paths = [os.path.join(depth_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]
    all_video_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_video_paths]
    all_flow_paths = [os.path.join(flow_dir, os.path.splitext(os.path.basename(f))[0]) for f in all_video_paths]

    if data_num != "all":
        data_num = int(data_num)
        all_video_paths = all_video_paths[:data_num]
        all_depth_paths = all_depth_paths[:data_num]
        all_frame_paths = all_frame_paths[:data_num]
        all_video_ids = all_video_ids[:data_num]
        all_flow_paths = all_flow_paths[:data_num]

    output_dir = args.output_dir / "step01_output"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        "allow_model_download": ALLOW_MODEL_DOWNLOAD,
        "output_dir": output_dir,
        "device": args.device,
        "video_path": all_video_paths,
        "frame_path": all_frame_paths,
        "depth_path": all_depth_paths,
        "flow_path": all_flow_paths,
        "video_ids": all_video_ids,
        # yolov8 world
        "od_model_path": root / args.driving_mini_od_model,
        "classes": args.driving_mini_obj_classes,
        "frame_rate":  args.bdd100k_frame_rate,
        "primary_confidence": args.primary_confidence,
        "candidate_confidence": args.candidate_confidence,
        "nms_iou": args.nms_iou,
        "inference_size": args.inference_size,
        # sam2
        "mask_model_path": root / args.sam2_model,
        "sam_prompt_candidates": args.sam_prompt_candidates,
        # Depth Anything v3
        "depth_model": args.depth_model,
        "depth_process_resolution": args.depth_process_resolution,
        # Flow model
        "flow_consistency_threshold_px": args.flow_consistency_threshold_px,
        
    }
    return input_data

def get_step_02_input(args):
    output_dir = args.output_dir / "step02_output"
    os.makedirs(output_dir, exist_ok=True)
    input_data = {
        "output_dir": output_dir,
        "device": args.device,
        "step01_output_dir": args.output_dir / "step01_output",
    }
    return input_data

