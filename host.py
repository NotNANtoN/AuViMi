import os
import sys
import time
import shutil
import argparse

import numpy as np
import torchvision
from PIL import Image

from utils import time_stamp, kill_old_process, clean_pid

sys.path.append("../deepdaze/")
from deep_daze_repo.deep_daze.deep_daze import Imagine


def clean_host_folders():
    shutil.rmtree('host_in')
    shutil.rmtree('host_out')
     
    
def timestr():
    return time.strftime("%x", time.gmtime())
    

kill_old_process(create_new=True)


# Do some actual work here
host_in = "host_in"
host_out = "host_out"
os.makedirs(host_in, exist_ok=True)
os.makedirs(host_out, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--gradient_accumulate_every", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=44)
args = parser.parse_args()

try:
    to_pil = torchvision.transforms.ToPILImage()
    
    train_steps = 1
    model = Imagine(
                epochs = args.epochs,
                image_width=args.size,
                gradient_accumulate_every=args.gradient_accumulate_every,
                batch_size=args.batch_size,
                num_layers=args.num_layers,
                
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
    # make videos
    folder = "results"
    os.makedirs(folder, exist_ok=True)
    time_now = timestr()
    path = os.path.join(folder, time_now)
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_out","%d.jpg"), "-pix_fmt", "yuv420p", path + "_mirror.mp4"])
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_in","%d.jpg"), "-pix_fmt", "yuv420p", path + "_input.mp4"])
    # clear folders
    clean_folders()
    # kill process
    clean_pid()


