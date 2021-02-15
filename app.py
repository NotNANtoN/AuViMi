import subprocess
import pipes

import cv2
from PIL import Image


host = "abakus"
repo_name = "AuViMi"


# make sure that repo is cloned on host
exists = subprocess.call(['ssh', host, 'test -e ' + pipes.quote(repo_name)]) == 0
print("Repo exists: ", exists)
if not exists:
    subprocess.run(['ssh', host, 'git', 'clone', 'git@github.com:NotNANtoN/AuViMi.git'])
else:
    subprocess.run(['ssh', host, 'cd AuViMi', ';', 'git', 'pull'])
# start host process
subprocess.run(['ssh', host, 'python3', 'AuViMi/host.py'])

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    cap.release()
    cap = cv2.VideoCapture(0)


while (cap.isOpened()):
    success, frame = cap.read()
    
    # return on escape
    if cv2.waitKey(33) == 27:
        success = False

    
    if success:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Video", rgb_frame)
    else:
        break
        
    
cap.release()

