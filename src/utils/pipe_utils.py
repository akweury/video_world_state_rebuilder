

import argparse
import os
import platform
import re
import sys
import time
from pathlib import Path

import config 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the VWSR pipeline"
    )
    
    parser.add_argument("--data", type=int)
    parser.add_argument("--data", type=int)
    return parser.parse_args()

