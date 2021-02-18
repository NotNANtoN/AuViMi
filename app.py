import os
import subprocess
import pipes
import time
import sys
import threading
import argparse
import joblib

import numpy as np
import cv2
import torchvision
from PIL import Image


from utils import time_stamp


def output_reader(proc):
    for line in iter(proc.stdout.readline, ''):
        print(line.decode('utf-8'), end='')
        #print('got line: {0}'.format(line.decode('utf-8')), end='')



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
        args_cli.append("--" + key)
        args_cli.append(str(args[key]))
    host_process = subprocess.Popen(commands + args_cli, stdout=subprocess.PIPE)
    
    #t = threading.Thread(target=output_reader, args=(host_process,))
    #t.start()
    #def get_output(host_process):
    #    for stdout_line in iter(host_process.stdout.readline, ""):
    #        yield stdout_line 
   
    #host_stdout_iterator = get_output(host_process)

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
    
    old_hash = None
    new_img_time = time.time()
    
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
        if cv2.waitKey(33) == 27:
            success = False

        if success:
            # move img over:
            img_name = str(count) + ".jpg"
            count += 1
            img_path = os.path.join(client_out, img_name)
            # save img
            frame.save(img_path, quality=95, subsampling=0)
            # send to host
            target_path = os.path.join(host_in, img_name)
            subprocess.Popen(['rsync', img_path, host_scp_path + total_path + target_path])
            # display img
            #rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            cv2.imshow("Input", frame_cv2)
            
            # get processed img (if there is a new one):
            new_img_name = "new.jpg"
            local_img_name = str(count) + ".jpg"
            host_path = os.path.join(host_out, new_img_name)
            client_path = os.path.join(client_in, new_img_name)
            
            if os.path.exists(client_path):
                # load processed img
                try:
                    img = np.array(Image.open(client_path))
                    #img = np.float32(np.load(client_path))
                    new_hash = joblib.hash(img)
                    if new_hash != old_hash:
                        old_hash = new_hash
                        print("Seconds between trainings: ", time.time() - new_img_time)
                        new_img_time = time.time()
                    # show processed img
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Mirror", img)
                except (ValueError, OSError):
                    pass
        
            # load new image asynchronously
            subprocess.Popen(['rsync', host_scp_path + total_path + host_path, client_path])
        else:
            break
            
        print("Time per client loop: ", time.time() - start_loop_time)
        
        #for line in host_stdout_iterator:
        #    print(line)
        #host_out = next(host_stdout_iterator)
        #print(host_out)
        
        
        #print(get_stdout(host_process))
           
    #t.join()
    cap.release()
    
    
    
if __name__ == "__main__":
    host = "abakus.ddnss.de"
    user = "anton"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=44)
    args = parser.parse_args()

    try:
        main(host, user, args)
    finally:
        subprocess.Popen(['ssh', host, 'python3', '~/AuViMi/stop_host.py'])
    


