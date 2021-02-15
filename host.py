import os
import sys
import time

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
                image_width=128,
                #save_progress=True,
                #open_folder=True,
                #start_image_train_iters=200,
               )



    previous_img = None
    newest_img = None
    count = 0
    while True:
        img_names = os.listdir(host_in)
        newest_img = max([int(name[-4]) for name in img_names]) if len(img_names) > 0 else 0
        
        # maybe update target img
        if newest_img != previous_img:
            model.set_clip_encoding(newest_img)
        # train one step
        img_tensor, loss = model.train_step()
        # save new img
        img_pil = to_pil(img_tensor.cpu())
        new_name = str(count) + ".png"
        img_pil.save(os.path.join(host_out, new_name))
        # next
        previous_img = newest_img

finally:
    clean_pid()


