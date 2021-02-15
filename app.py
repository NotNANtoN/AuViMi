import os
import subprocess
import pipes

import numpy as np
import cv2
import torchvision
from PIL import Image


from utils import time_stamp


def main(host, user):
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
    subprocess.Popen(['ssh', host, 'cd AuViMi;', host_python_path, 'host.py'])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # we only want to store the newsest img
    if (cap.isOpened() == False):
        cap.release()
        cap = cv2.VideoCapture(0)

    to_pil = torchvision.transforms.ToPILImage()

    newest = 0
    previous = 0
    count = 0
    while(cap.isOpened()):
        success, frame_np = cap.read()
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
            img_name = str(count) + ".png"
            count += 1
            img_path = os.path.join(client_out, img_name)
            target_path = os.path.join(host_in, img_name)
            #np.save(img_path, rgb_frame)
            frame.save(img_path)
           
            subprocess.run(['scp', img_path, host_scp_path + total_path + target_path])
            # display img
            rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            cv2.imshow("Input", rgb_frame)
            
            # get processed img (if there is a new one):
            # check get list of imgs
            process_out = subprocess.run(['ssh', host, 'ls ' + total_path + host_out], stdout=subprocess.PIPE)
            all_host_outs = process_out.stdout.decode("utf-8").split("\n")[:-1]
            newest = max([int(name[:-4]) for name in all_host_outs]) if len(all_host_outs) > 0 else 0
            if newest > previous:
                new_img_name = str(newest) + ".png"
                host_path = os.path.join(host_out, new_img_name)
                client_path = os.path.join(client_in, new_img_name)
                subprocess.run(['scp', host_scp_path + total_path + host_path, client_path])
                # load processed img
                processed_img = Image.open(client_path)
                # show processed img
                cv2.imshow("Mirror", processed_img)
                previous = newest
                
        else:
            break
              
    cap.release()
    
    
    
if __name__ == "__main__":
    host = "abakus.ddnss.de"
    user = "anton"
    try:
        main(host, user)
    finally:
        subprocess.Popen(['ssh', host, 'python3', '~/AuViMi/stop_host.py'])
    


