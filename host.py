import os
import sys
import time

from utils import time_stamp, kill_old_process

sys.path.append("../deepdaze/deep_daze_repo/deep_daze")
print(sys.path)
from deep_daze.deep_daze import Imagine


kill_old_process(create_new=True)


# Do some actual work here
host_in = "host_in"
host_out = "host_out"
os.makedirs(host_in exist_ok=True)
os.makedirs(host_out exist_ok=True)

newest_img = None
while True:
    images = os.listdir(img_folder)
    print(images)
    time.sleep(2)

