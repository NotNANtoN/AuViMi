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

# import experimental repositories if on abakus
if args.host == "abakus.ddnss.de":
    print(vars(args))
    if args.gen_backbone == "deepdaze":
        sys.path.append("../deepdaze/")
        from deep_daze_repo.deep_daze.deep_daze import Imagine   
    elif args.gen_backbone == "bigsleep"::
        sys.path.append("../")
        from big_sleep_repo.big_sleep.big_sleep import Imagine
    elif args.gen_backbone == "styleclip":
        sys.path.append("../")
        from StyleCLIP_modular.style_clip import Imagine
else:
    if args.gen_backbone == "deepdaze":
        from deep_daze import Imagine   
    elif args.gen_backbone == "bigsleep":
        from big_sleep import Imagine
    elif args.gen_backbone == "styleclip":
        from style_clip import Imagine


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
                    lr=args.lr,
                    lower_bound_cutout=args.lower_bound_cutout,                
                    open_folder=False,
                    save_progress=False,
                    center_bias=args.center_bias,
                    hidden_size=args.hidden_size,
                    averaging_weight=args.averaging_weight,
                   )
    elif args.gen_backbone == "styleclip":
        model = Imagine(
                    style=args.style,
                    epochs = args.epochs,
                    image_width=args.size,
                    gradient_accumulate_every=args.gradient_accumulate_every,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lower_bound_cutout=args.lower_bound_cutout,                
                    open_folder=False,
                    save_progress=False,
                    center_bias=args.center_bias,
                    averaging_weight=args.averaging_weight,
                   )
    elif args.gen_backbone == "bigsleep"::
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
            # reptile(openai)/FOMAML(Finn) approach
            slow_weights = model.state_dict().copy()
            # update fast_weight for n steps
            for _ in range(args.opt_steps):
                img_tensor, loss = model.train_step(0, count)
            adapted_weights = model.state_dict()
            # take the slow_weights a step closer to the updated fast_weights 
            # pseudoversion: new_slow_weights = slow_weights + args.meta_lr * (adapted_weights - slow_weights)
            for key in slow_weights:
                new_slow_weights = slow_weights[key] + args.meta_lr * (adapted_weights[key] - slow_weights[key])
                slow_weights[key] = new_slow_weights.type(slow_weights[key].dtype)
            # put the updated slow weights back in the model
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
    # save output movie
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_out","%d.jpg"), "-pix_fmt", "yuv420p", path + "_mirror.mp4"])
    # rename host_in images for ffmpeg:
    files = os.listdir("host_in")
    files = sorted(files, key=lambda f: int(f[:-4]))
    for f, i in zip(files, range(len(files))):
        orig_name = os.path.join("host_in", f)
        new_name = os.path.join("host_in", str(i) + ".jpg")
        if orig_name != new_name:
            os.rename(orig_name, new_name)
    # save input movie
    subprocess.run(["ffmpeg", "-i", os.path.join(os.getcwd(), "host_in","%d.jpg"), "-pix_fmt", "yuv420p", path + "_input.mp4"])
    # clean folders
    #clean_host_folders()
    # kill process
    clean_pid()


