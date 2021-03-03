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
    repo_name = "AuViMi"
    total_path = "~/AuViMi/"
    client_out = os.path.join("client_out")
    client_in = os.path.join("client_in")
    host_in = os.path.join("host_in")
    host_out = os.path.join("host_out")
    host_python_path = "/home/anton/anaconda3/bin/python"
    host_scp_path = user + "@" + host + ":"
    
    resize_size = 224

    os.makedirs(client_out, exist_ok=True)
    os.makedirs(client_in, exist_ok=True)
    
    new_img_name = "new.jpg"
    host_path = os.path.join(host_out, new_img_name)
    client_path = os.path.join(client_in, new_img_name)
    
    clean_client_folders()

    # make sure that repo is cloned on host
    exists = subprocess.call(['ssh', host, 'test -e ' + pipes.quote(repo_name)]) == 0
    print("Repo exists: ", exists)
    if not exists:
        subprocess.run(['ssh', host, 'git', 'clone', 'git@github.com:NotNANtoN/AuViMi.git'])
    else:
        subprocess.run(['ssh', host, 'cd AuViMi;', 'git', 'pull'])

    # start host process
    commands = ['ssh', host, 'cd AuViMi;', host_python_path, 'host.py']
    args_cli = []
    args = vars(args)
    for key in args:
        value = str(args[key])
        if key == "text":
            if value is None or value == "":
                continue
            value = '"' + value + '"'
        args_cli.append("--" + key)
        args_cli.append(value)
    output_cmds = [">", "cli_out.txt"]
    host_process = subprocess.Popen(commands + args_cli + output_cmds, stdout=subprocess.PIPE)

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
            subprocess.Popen(['rsync', img_path, host_scp_path + total_path + target_path])
            # display img
            #rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            if args["mode"] == "pic":
                move_pic = False
                cv2.imshow("Optimization goal", frame_cv2)
        
        if args["text_weight"] != 1.0:
            cv2.imshow("Input", frame_cv2)
        
        # get processed img (if there is a new one):
        local_img_name = str(count) + ".jpg"
        
        # load new image asynchronously
        subprocess.Popen(['rsync', host_scp_path + total_path + host_path, client_path])
        
        if os.path.exists(client_path):
            # load processed img
            try:
                pil_img_small = Image.open(client_path)
                np_img_small = np.array(pil_img_small)
                pil_img_large = pil_img_small.resize((512, 512))
                np_img_large = np.array(pil_img_large)

                new_hash = joblib.hash(np_img_small)
                if new_hash != old_hash:
                    old_hash = new_hash
                    host_timing = time.time() - new_img_time
                    print("Seconds between trainings: ", host_timing)
                    host_timings.append(host_timing)
                    new_img_time = time.time()
                # show processed img
                img = cv2.cvtColor(np_img_large, cv2.COLOR_RGB2BGR)
                cv2.imshow("Mirror", np_img_large)
            except (ValueError, OSError):
                pass
        
            
        client_timing = time.time() - start_loop_time
        #print("Time per client loop: ", client_timing)
        client_timings.append(client_timing)
        
        #for line in host_stdout_iterator:
        #    print(line)
        #host_out = next(host_stdout_iterator)
        #print(host_out)
        
        
        #print(get_stdout(host_process))
           
    print("Mean client time:", np.mean(client_timings))
    print("Mean host time:", np.mean(host_timings))
    #t.join()
    cap.release()
    return host_process
    
    
if __name__ == "__main__":  
    args = get_args()
    
    host = "abakus.ddnss.de"
    user = "anton"

    try:
        host_process = main(host, user, args)
        #host_process.send_signal(signal.SIGINT)
    finally:
        #subprocess.Popen(['ssh', host, 'python3', '~/AuViMi/stop_host.py'])
        # create file to tell process to stop!
        subprocess.Popen(['ssh', host, 'touch', '~/AuViMi/STOP.txt'])
    


