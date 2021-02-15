import os
import signal
import time


def time_stamp():
    return time.strftime("%X", time.gmtime())
    
    
def kill_old_process(create_new=False):
    pid = str(os.getpid())
    pidfile = "/tmp/mydaemon.pid"
    if os.path.isfile(pidfile):
        print("Process already exists, killing old process...")
        with open(pidfile, 'r') as f:
            old_pid = f.read()
        print("Old PID: ", old_pid)
        os.kill(old_pid, signal.SIGTERM)
        time.sleep(2)
    if create_new:
        with open(pidfile, 'w') as f:
            f.write(pid)
