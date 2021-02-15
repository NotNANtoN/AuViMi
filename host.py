import os
import sys
import joblib
import time


pid = str(os.getpid())
pidfile = "/tmp/mydaemon.pid"
if os.path.isfile(pidfile):
    print("Process already exists, killing old process...")
    old_pid = joblib.load(pidfile)
    os.kill(old_pid, singal.SIGTERM)
    time.sleep(2)
joblib.dump(pid, pidfile)

try:
    # Do some actual work here
    img_folder = "host_input"
    os.makedirs(image_folder, exist_ok=True)
    newest_img = None
    while True:
        images = os.listdir(img_folder)
        print(images)<
        time.sleep(2)
    
finally:
    os.unlink(pidfile)
