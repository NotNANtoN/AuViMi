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
    subprocess.Popen(['ssh', host, 'cd AuViMi;', 'python3', 'AuViMi/host.py'])

    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        cap.release()
        cap = cv2.VideoCapture(0)


    while(cap.isOpened()):
        success, frame = cap.read()
        
        # return on escape
        if cv2.waitKey(33) == 27:
            success = False

        
        if success:
            # move img over:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_name = time_stamp()
            img_path = os.path.join(client_out, img_name)
            target_path = os.path.join(repo_name, host_in, img_name)
            np.save(img_path, rgb_frame)
            #rgb_frame.save(img_path)
            subprocess.run(['scp', host, img_path, target_path])
            # display img
            cv2.imshow("Video", rgb_frame)
            
            # get processed img (if there is a new one):
            #subprocess.run(['scp', host, img_path, target_path])
        else:
            break
              
    cap.release()
    
    
    
if __name__ == "__main__":
    host = "abakus"
    try:
        main(host)
    finally:
        subprocess.Popen(['ssh', host, 'python3', 'AuViMi/stop_host.py'])
    


