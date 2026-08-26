# Created by MacBook Pro at 21.08.26

"""
# the pipeline has the following steps:
# step 1: detect objects masks, labels, depths in
# each frame of the video, cache the results as intermediate files
# step 2: estimate the 3D pose of each object in each frame.

# step 3: rebuild the world state of the camera and the objects

# step 4: reasoning the conflict of the world states based
# on the different fact sources

# step 5: re-estimate the world state with less conflicts

# step 6: comparing with the ground truth,
# check the accuracy of the world state estimation.
"""

import numpy as np


from src.utils import pipe_utils
from src.pipeline import step01, step02
from src import config



def main():
    args = pipe_utils.parse_args()
    # step 0: system check, and set up the environment
    args = config.step_0_setup(args)
    # step 1: detect objects masks, labels, depths in
    # each frame of the video, cache the results as intermediate files
    step01_input = config.get_step_01_input(args)
    step01.main(step01_input)

    # step 2: estimate the 3D pose of each object in each frame.
    step02_input = config.get_step_02_input(args)
    step02.main(step02_input)
    return   


if __name__ == "__main__":
    print("\n------- Start the pipeline -------\n")
    res = main()
    print("\n ------- Program Finished! ------ \n")