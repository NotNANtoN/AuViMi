import os
import sys
import time

import numpy as np
import torchvision
from PIL import Image

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
    
    train_steps = 1
    model = Imagine(
                epochs = 12,
                image_width=256,
                gradient_accumulate_every=1,
                batch_size=8,
                num_layers=40,
                
                #lr=3e-3   # 3e-3 is unstable
                
                #save_progress=True,
                #open_folder=True,
                #start_image_train_iters=200,
               )

    # delete previous imgs
    for f in os.listdir(host_in):
        os.unlink(os.path.join(host_in, f))
    for f in os.listdir(host_out):
        os.unlink(os.path.join(host_out, f))

    previous_img = None
    newest_img = None
    count = 0
    while True:
        host_loop_time = time.time()
        img_names = os.listdir(host_in)
        img_names = [name for name in img_names if name.endswith(".jpg")]
        newest_img = max([int(name[:-4]) for name in img_names]) if len(img_names) > 0 else 0
        
        # maybe update target img
        if newest_img != previous_img:
            img_path = os.path.join(host_in, str(newest_img) + ".jpg")
            model.set_clip_encoding(img=img_path)
            previous_img = newest_img
        # train
        start_train_time = time.time()
        for _ in range(train_steps):
            img_tensor, loss = model.train_step(0, count)

        # save new img
        img_np = np.uint8(img_tensor.cpu().detach().squeeze().permute(1, 2, 0).numpy() * 255)
        img_pil = Image.fromarray(img_np)
        count += 1
        img_pil.save(os.path.join(host_out, "new.jpg"), quality=95, subsampling=0)
        img_pil.save(os.path.join(host_out, str(count) + ".jpg"), quality=95, subsampling=0)
        #np.save(os.path.join(host_out, "new.npy"), img_np)
        #np.save(os.path.join(host_out, str(count) + ".npy"), img_np)
        



finally:
    clean_pid()


