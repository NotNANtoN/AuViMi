import os
import subprocess
import pipes

import numpy as np
import cv2
from PIL import Image

from utils import time_stamp

def main(host):
    repo_name = "AuViMi"
    image_folder = "client_out"
    processed_folder = "client_in"
    host_images = "host_in"
    host_processed = "host_out"

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)


    # make sure that repo is cloned on host
    exists = subprocess.call(['ssh', host, 'test -e ' + pipes.quote(repo_name)]) == 0
    print("Repo exists: ", exists)
    if not exists:
        subprocess.run(['ssh', host, 'git', 'clone', 'git@github.com:NotNANtoN/AuViMi.git'])
    else:
        subprocess.run(['ssh', host, 'cd AuViMi', ';', 'git', 'pull'])
    # start host process
    subprocess.Popen(['ssh', host, 'python3', 'AuViMi/host.py'])

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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_name = time_stamp() + ".png"
            img_path = os.path.join(image_folder, img_name)
            target_path = os.path.join(repo_name, host_images, img_name)
            np.save(img_path, rgb_frame)
            #rgb_frame.save(img_path)
            subprocess.run(['scp', host, img_path, target_path])
            cv2.imshow("Video", rgb_frame)
        else:
            break
              
    cap.release()
    
    
    
if __name__ == "__main__":
    host = "abakus"
    try:
        main(host)
    finally:
        subprocess.Popen(['ssh', host, 'python3', 'AuViMi/stop_host.py'])
    


