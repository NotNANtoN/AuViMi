import os
import sys
import time

import numpy as np
import torchvision

from utils import time_stamp, kill_old_process, clean_pid

sys.path.append("../deepdaze/")
from deep_daze_repo.deep_daze.deep_daze import Imagine

    

kill_old_process(create_new=True)


# Do some actual work here
host_in = "host_in"
host_out = "host_out"
os.makedirs(host_in, exist_ok=True)
os.makedirs(host_out, exist_ok=True)


try:
    to_pil = torchvision.transforms.ToPILImage()
    
    model = Imagine(
                epochs = 12,
                image_width=512,
                #save_progress=True,
                #open_folder=True,
                #start_image_train_iters=200,
               )

    previous_img = None
    newest_img = None
    count = 0
    while True:
        img_names = os.listdir(host_in)
        img_names = [name for name in img_names if name.endswith(".png")]
        newest_img = max([int(name[:-4]) for name in img_names]) if len(img_names) > 0 else 0
        
        # maybe update target img
        if newest_img != previous_img:
            png_path = os.path.join(host_in, str(newest_img) + ".png")
            model.set_clip_encoding(img=png_path)
            previous_img = newest_img
        # train one step
        img_tensor, loss = model.train_step(0, count)
        # save new img
        img_np = img_tensor.cpu().detach().squeeze().numpy()
        count += 1
        np.save(os.path.join(host_out, "new.npy"), img_np)
        np.save(os.path.join(host_out, str(count) + ".npy"), img_np)


finally:
    clean_pid()


