import os
import sys
import time

from utils import time_stamp, kill_old_process

sys.path.append("../deepdaze/deep_daze_repo/deep_daze")
print(sys.path.list())
from deep_daze.deep_daze import Imagine


kill_old_process(create_new=True)


# Do some actual work here
img_folder = "host_in"
out_folder = "host_out"
os.makedirs(img_folder, exist_ok=True)
newest_img = None
while True:
    images = os.listdir(img_folder)
    print(images)
    time.sleep(2)

