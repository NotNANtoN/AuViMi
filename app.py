import os
import subprocess
import pipes
import time
import sys
import threading
import joblib
import signal

import numpy as np
import cv2
import torchvision
from PIL import Image
import torch
import torch.nn.functional as F


from utils import time_stamp, get_args, clean_folder


def output_reader(proc):
    for line in iter(proc.stdout.readline, ''):
        print(line.decode('utf-8'), end='')
        #print('got line: {0}'.format(line.decode('utf-8')), end='')


def clean_client_folders():
    clean_folder("client_in")
    clean_folder("client_out")


def main(host, user, args):

    # small fix for myself
    if args.host == "abakus.ddnss.de" and not args.run_local:
        args.python_path = "/home/anton/anaconda3/bin/python"

    repo_name = "AuViMi"
    total_path = "~/AuViMi/"
    client_out = os.path.join("client_out")
    client_in = os.path.join("client_in")
    host_in = os.path.join("host_in")
    host_out = os.path.join("host_out")
    host_python_path = args.python_path
    host_scp_path = user + "@" + host + ":"
    
    resize_size = 224

    # create folders
    for f in (client_out, client_in, host_out, host_in):
        os.makedirs(f, exist_ok=True)
    
    new_img_name = "new.jpg"
    host_path = os.path.join(host_out, new_img_name)
    client_path = os.path.join(client_in, new_img_name)
    
    clean_client_folders()


    # make sure that repo is cloned on host
    if not args.run_local:
        exists = subprocess.call(['ssh', host, 'test -e ' + pipes.quote(repo_name)]) == 0
        print("Repo exists: ", exists)
        if not exists:
            subprocess.run(['ssh', host, 'git', 'clone', 'git@github.com:NotNANtoN/AuViMi.git'])
            subprocess.run(['ssh', host, 'cd AuViMi;', host_python_path, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        else:
            subprocess.run(['ssh', host, 'cd AuViMi;', 'git', 'pull'])

    # filter irrelevant args
    args = vars(args)
    if args["gen_backbone"] == "bigsleep":
        del args["num_layers"]
        del args["lr"]
        del args["saturate_bound"]
        del args["lower_bound_cutout"]
        del args["center_bias"]
        del args["center_focus"]
        del args["hidden_size"]
        del args["averaging_weight"]

    # start host process
    if args["run_local"]:
        commands = [host_python_path, 'host.py']
    else:
        commands = ['ssh', host, 'cd AuViMi;', host_python_path, 'host.py']
    args_cli = ["--" + key + '="' + str(args[key]) + '"' for key in args if (args[key] is not None and str(args[key]) != "")]
    output_cmds = [">", "cli_out.txt"]
    all_cmds = commands + args_cli + output_cmds
    joined = " ".join(all_cmds)
    if args["run_local"]:
        host_process = subprocess.Popen(joined, shell=True, stdout=subprocess.PIPE)
    else:
        host_process = subprocess.Popen(all_cmds, shell=False, stdout=subprocess.PIPE)

    # init webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # we only want to store the newest img
    if (cap.isOpened() == False):
        cap.release()
        cap = cv2.VideoCapture(0)

    to_pil = torchvision.transforms.ToPILImage()

    newest = 0
    previous = 0
    count = 0
    
    move_pic = True if args["mode"] == "stream" else False
    old_hash = None
    new_img_time = time.time()
    
    client_timings = []
    host_timings = []
    
    rsync_cmds = ["rsync", "-chazuq", "--ignore-missing-args"]
    
    while(cap.isOpened()):
        start_loop_time = time.time()
        success, frame_cv2 = cap.read()
        
        
        frame_np = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        frame = to_pil(frame_np).convert("RGB")
        
        # resize to have smaller transfer
        x, y = frame.size
        if x > y:
            y_target = resize_size
            x_target = int(x / y * y_target)
        else:
            x_target = resize_size
            y_target = int(y / x * x_target)   
        frame = frame.resize((x_target, y_target))
        
        # return on escape
        key = cv2.waitKey(1)
        if key == 27:
            print("Pressed Escape. Quitting!")
            break
            
        if args["mode"] == "pic" and key == ord("p"):
            move_pic = True

        if move_pic == True:
            # move img over:
            img_name_host = str(count) + ".jpg"
            img_name_client = img_name_host #"new.jpg"  

            count += 1
            img_path = os.path.join(client_out, img_name_client)
            # save img
            frame.save(img_path, quality=95, subsampling=0)
            # send to host
            target_path = os.path.join(host_in, img_name_host)

            # display img
            #rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            if args["mode"] == "pic":
                transfer_cmd = subprocess.run
                move_pic = False
                cv2.imshow("Optimization goal", frame_cv2)
            else:
                transfer_cmd = subprocess.Popen
            if args["run_local"]:
                frame.save(target_path, quality=95, subsampling=0)
            else:
                transfer_cmd(rsync_cmds + [img_path, host_scp_path + total_path + target_path])
        
        if args["text_weight"] != 1.0:
            cv2.imshow("Input", frame_cv2)

        
        # load new image from host asynchronously
        if not args["run_local"]:
            subprocess.Popen(rsync_cmds + [host_scp_path + total_path + host_path, client_path])
        
        if os.path.exists(client_path):
            # load processed img
            try:
                pil_img_small = Image.open(client_path)
                np_img_small = np.array(pil_img_small)

                new_hash = joblib.hash(np_img_small)
                if new_hash != old_hash:
                    old_hash = new_hash
                    host_timing = time.time() - new_img_time
                    #print("Seconds between trainings: ", host_timing)
                    host_timings.append(host_timing)
                    new_img_time = time.time()
                else:
                    continue
                pil_img_large = pil_img_small.resize((512, 512))
                np_img_large = np.array(pil_img_large)
                # convert to cv2 color space
                img = cv2.cvtColor(np_img_large, cv2.COLOR_RGB2BGR)
                cv2.imshow("Mirror", img)
            except (ValueError, OSError):
                pass
            
        client_timing = time.time() - start_loop_time
        #print("Time per client loop: ", client_timing)
        client_timings.append(client_timing)
           
    print("Mean client time:", np.mean(client_timings))
    print("Mean host time:", np.mean(host_timings))
    cap.release()
    return host_process
    
    
if __name__ == "__main__":  
    args = get_args()
    
    host = "abakus.ddnss.de"
    user = "anton"

    try:
        host_process = main(host, user, args)
    finally:
        # create file to tell process to stop!
        if not args.run_local:
            subprocess.Popen(['ssh', host, 'touch', '~/AuViMi/STOP.txt'])
    


