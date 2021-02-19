import os
import sys
import time
import shutil
import argparse
import subprocess

import numpy as np
import torchvision
from PIL import Image

from utils import time_stamp, kill_old_process, clean_pid, get_args, clean_folder

sys.path.append("../deepdaze/")
from deep_daze_repo.deep_daze.deep_daze import Imagine



def clean_host_folders():
    clean_folder("host_in")
    clean_folder("host_out")
    
def timestr():
    return time.strftime("%x_%X", time.gmtime()).replace("/", "_")
    

kill_old_process(create_new=True)
if os.path.exists("STOP.txt"):
        os.unlink("STOP.txt")

# Do some actual work here
host_in = "host_in"
host_out = "host_out"
os.makedirs(host_in, exist_ok=True)
os.makedirs(host_out, exist_ok=True)


args = get_args()

clean_host_folders()

try:
    to_pil = torchvision.transforms.ToPILImage()
    
    model = Imagine(
                epochs = args.epochs,
                image_width=args.size,
                gradient_accumulate_every=args.gradient_accumulate_every,
                batch_size=args.batch_size,
                num_layers=args.num_layers,
                lr=args.lr,   # 3e-3 is unstable
                
                open_folder=False,
                #start_image_train_iters=200,
               )

    text_encoding = None
    img_encoding = None
    if args.text is not None and args.text != "":
        text_encoding = model.create_text_encoding(args.text)
    text_weight = args.text_weight
    previous_img = None
    newest_img = None
    count = 0
    
    
    while not os.path.exists("STOP.txt"):
        host_loop_time = time.time()
        
        img_names = [name[:-4] for name in os.listdir(host_in) if name.endswith(".jpg")]
        newest_img = max(img_names, key=lambda x: int(x)) if len(img_names) > 0 else 0
        
        # maybe update target img
        if newest_img != previous_img:
            img_path = os.path.join(host_in, str(newest_img) + ".jpg")
            new_img_encoding = model.create_img_encoding(img_path) if text_weight < 1.0 else text_encoding
            # update running avg of img encoding
            if img_encoding is None:
                img_encoding = new_img_encoding
            else:
                img_encoding = args.run_avg * img_encoding + (1 - args.run_avg) * img_encoding
            # merge text and img encoding
            if text_encoding is None:
                clip_encoding = img_encoding
            else:
                clip_encoding = img_encoding * (1 - text_weight) + text_encoding * text_weight
            clip_encoding /= clip.norm(dim=-1, keepdim=True)
            model.set_clip_encoding(encoding=clip_encoding)
            previous_img = newest_img
        # train
        start_train_time = time.time()
        for _ in range(args.opt_steps):
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
    if os.path.exists("STOP.txt"):
        os.unlink("STOP.txt")
    # make videos
    folder = "results"
    os.makedirs(folder, exist_ok=True)
    time_now = timestr()
    path = os.path.join(os.getcwd(), folder, time_now)
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_out","%d.jpg"), "-pix_fmt", "yuv420p", path + "_mirror.mp4"])
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_in","%d.jpg"), "-pix_fmt", "yuv420p", path + "_input.mp4"])
    # clean folders
    #clean_host_folders()
    # kill process
    clean_pid()


