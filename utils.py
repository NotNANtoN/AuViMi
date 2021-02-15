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
        try:
            os.kill(int(old_pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.unlink(pidfile)
        time.sleep(2)
    if create_new:
        with open(pidfile, 'w') as f:
            f.write(pid)


def clean_pid():
    pidfile = "/tmp/mydaemon.pid"
    os.unlink(pidfile)

