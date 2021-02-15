import os
import sys
import time

from utils import time_stamp, kill_old_process

sys.path.append("../deepdaze/deep_daze_repo")
from deep_daze.deep_daze import Imagine


kill_old_process(create_new=True)


# Do some actual work here
img_folder = "host_input"
out_folder = "host_output"
os.makedirs(img_folder, exist_ok=True)
newest_img = None
while True:
    images = os.listdir(img_folder)
    print(images)
    time.sleep(2)

