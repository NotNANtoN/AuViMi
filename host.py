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
client_in = "client_in"
os.makedirs(client_in, exist_ok=True)
os.makedirs(host_in, exist_ok=True)
os.makedirs(host_out, exist_ok=True)
os.makedirs("debug", exist_ok=True)

args = get_args()

if args.host == "abakus.ddnss.de":
    if args.gen_backbone == "deepdaze":
        sys.path.append("../deepdaze/")
        from deep_daze_repo.deep_daze.deep_daze import Imagine   
    else:
        sys.path.append("../")
        from big_sleep_repo.big_sleep.big_sleep import Imagine
else:
    if args.gen_backbone == "deepdaze":
        from deep_daze import Imagine   
    else:
        from big_sleep import Imagine


clean_host_folders()

try:
    to_pil = torchvision.transforms.ToPILImage()
    
    if args.gen_backbone == "deepdaze":
        model = Imagine(
                    epochs = args.epochs,
                    image_width=args.size,
                    gradient_accumulate_every=args.gradient_accumulate_every,
                    batch_size=args.batch_size,
                    num_layers=args.num_layers,
                    lr=args.lr,   # 3e-3 is unstable
                    lower_bound_cutout=args.lower_bound_cutout,                
                    open_folder=False,
                    #start_image_train_iters=200,
                    save_progress=False,
                    do_occlusion=args.do_occlusion,
                    center_bias=args.center_bias,
                   )
    else:
        model = Imagine(
                save_progress=False,
                save_best=False,
                open_folder=False,
                num_cutouts=args.batch_size,
                image_size=args.size,
                epochs=args.epochs,
                gradient_accumulate_every=1,
               )

    text_weight = args.text_weight
    img_encoding = 0
    text_encoding = None
    if args.text is not None and args.text != "":
        print("Optimizing on ", args.text)
        text_encoding = model.create_text_encoding(args.text)
        text_encoding /= text_encoding.norm(dim=-1, keepdim=True)
    clip_encoding = text_encoding
    if text_weight == 1.0:
        model.set_clip_encoding(encoding=text_encoding)

    previous_img = None
    newest_img = None
    count = 0
    
    
    while not os.path.exists("STOP.txt"):
        host_loop_time = time.time()
        
        img_names = [name[:-4] for name in os.listdir(host_in) if name.endswith(".jpg")]
        newest_img = max(img_names, key=lambda x: int(x)) if len(img_names) > 0 else None
        
        # maybe update target img
        if text_weight < 1.0 and newest_img != previous_img:
            # determine img encoding
            img_path = os.path.join(host_in, str(newest_img) + ".jpg")
            print("updated img target: ", img_path)
            new_img_encoding = model.create_img_encoding(img_path)
            img_encoding = args.run_avg * img_encoding + (1 - args.run_avg) * new_img_encoding
            # merge image and text depending on conditions
            if text_encoding is None:
                clip_encoding = img_encoding
            else:
                clip_encoding = img_encoding * (1 - text_weight) + text_encoding * text_weight

            model.set_clip_encoding(encoding=clip_encoding)
            previous_img = newest_img
        if clip_encoding is None:
            continue
            
            
        
        # train
        if args.meta:
            # reptile approach (openai)
            slow_weights = model.state_dict().copy()
            
            for _ in range(args.opt_steps):
                img_tensor, loss = model.train_step(0, count)
            adapted_weights = model.state_dict()
            
            # pseudo: new_slow_weights = slow_weights + args.meta_lr * (adapted_weights - slow_weights)
            for key in slow_weights:
                new_slow_weights = slow_weights[key] + args.meta_lr * (adapted_weights[key] - slow_weights[key])
                slow_weights[key] = new_slow_weights.type(slow_weights[key].dtype)
            
            model.load_state_dict(slow_weights)    
            
        else:
            for _ in range(args.opt_steps):
                img_tensor, loss = model.train_step(0, count)

        # save new img
        img_np = np.uint8(img_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy() * 255)
        img_pil = Image.fromarray(img_np)
        count += 1
        img_pil.save(os.path.join(host_out, "new.jpg"), quality=95, subsampling=0)
        img_pil.save(os.path.join(host_out, str(count) + ".jpg"), quality=95, subsampling=0)
        if args.run_local:
            img_pil.save(os.path.join(client_in, "new.jpg"), quality=95, subsampling=0)
        

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


