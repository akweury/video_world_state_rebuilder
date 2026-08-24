

def main(input_data):
    print("\n------- Step 02 -------\n")
    output_dir = input_data["output_dir"]
    device = input_data["device"]
    step01_output_dir = input_data["step01_output_dir"]

    # Load the step 1 output data
    step01_output_data = load_step01_output(step01_output_dir)

    # Process each frame to estimate 3D pose
    for frame_data in step01_output_data:
        frame_id = frame_data["frame_id"]
        objects = frame_data["objects"]
        masks = frame_data["masks"]
        depths = frame_data["depths"]

        # Estimate 3D pose for each object
        for obj, mask, depth in zip(objects, masks, depths):
            pose_3d = estimate_3d_pose(obj, mask, depth, device)
            save_pose(output_dir, frame_id, obj["id"], pose_3d)




    print(f"Step 2 completed. 3D poses saved in {output_dir}")