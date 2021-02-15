import os
import subprocess
import pipes

import numpy as np
import cv2
from PIL import Image

from utils import time_stamp

def main(host):
    repo_name = "AuViMi"
    client_out = "client_out"
    client_int = "client_in"
    host_in = "host_in"
    host_processed = "host_out"
    host_python_path = "/home/anton/anaconda3/bin/python"
    
    resize_size = 224

    os.makedirs(client_out, exist_ok=True)
    os.makedirs(client_int, exist_ok=True)


    # make sure that repo is cloned on host
    exists = subprocess.call(['ssh', host, 'test -e ' + pipes.quote(repo_name)]) == 0
    print("Repo exists: ", exists)
    if not exists:
        subprocess.run(['ssh', host, 'git', 'clone', 'git@github.com:NotNANtoN/AuViMi.git'])
    else:
        subprocess.run(['ssh', host, 'cd AuViMi;', 'git', 'pull'])
    # start host process
    subprocess.Popen(['ssh', host, 'cd AuViMi;', host_python_path, 'host.py'])

    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        cap.release()
        cap = cv2.VideoCapture(0)


    count = 0
    while(cap.isOpened()):
        success, frame = cap.read()
        # resize to have smaller transfer
        x, y = frame.size()
        print(x, y)
        if x > y:
            y_target = resize_size
            x_target = int(x / y * y_target)
        else:
            x_target = resize_size
            y_target = int(y / x * x_target)   
        frame = frame.resize((x_target, y_target)
        
        # return on escape
        if cv2.waitKey(33) == 27:
            success = False

        if success:
            # move img over:
            img_name = "new" #str(count)
            count += 1
            img_path = os.path.join(client_out, img_name)
            target_path = os.path.join('~', repo_name, host_in, img_name)
            #np.save(img_path, rgb_frame)
            frame.save(img_path)
            subprocess.run(['scp', host, img_path, target_path])
            # display img
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Input", rgb_frame)
            
            # get processed img (if there is a new one):
            host_path = os.path.join(host_out, "new.png")
            received_img_name = str(count) + ".png"
            client_path = os.path.join(client_in, received_img_name)
            subprocess.run(['scp', host, host_path, client_path])
            # load processed img
            processed_img = Image.open(client_path)
            # show processed img
            cv2.imshow("Mirror", processed_img)
        else:
            break
              
    cap.release()
    
    
    
if __name__ == "__main__":
    host = "abakus"
    try:
        main(host)
    finally:
        subprocess.Popen(['ssh', host, 'python3', '~/AuViMi/stop_host.py'])
    


