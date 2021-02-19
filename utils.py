import os
import signal
import time
import argparse


def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=int, default=256)
        parser.add_argument("--epochs", type=int, default=12)
        parser.add_argument("--gradient_accumulate_every", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_layers", type=int, default=44)
        parser.add_argument("--host", type=str, default="abakus.ddnss.de")
        parser.add_argument("--user", type=str, default="anton")
        parser.add_argument("--text", type=str, default="")
        args = parser.parse_args()
        return args


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
            os.kill(int(old_pid), signal.SIGINT)
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

