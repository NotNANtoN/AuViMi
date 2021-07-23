import os
import signal
import time
import argparse


def get_args():
        parser = argparse.ArgumentParser()
        #parser.add_argument("--gen_backbone", default="deepdaze", choices=["deepdaze", "bigsleep", "styleclip"])
        parser.add_argument("--model_type", default="vqgan", choices=["siren", "vqgan", "conv", "raw"])
        parser.add_argument("--sideX", type=int, default=256)
        parser.add_argument("--sideY", type=int, default=256)
        parser.add_argument("--epochs", type=int, default=12)
        parser.add_argument("--gradient_accumulate_every", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=0.1)#1e-5)

        # styleclip
        parser.add_argument("--style", type=str, default="../stylegan2-ada-pytorch/VisionaryArt.pkl")
        # deepdaze
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--num_layers", type=int, default=32)
        # deepdaze + styleclip
        parser.add_argument("--saturate_bound", type=int, default=0)
        parser.add_argument("--lower_bound_cutout", type=float, default=0.05)
        #parser.add_argument("--do_occlusion", type=int, default=0)
        parser.add_argument("--center_bias", type=int, default=1)
        parser.add_argument("--center_focus", type=int, default=2)
        parser.add_argument("--averaging_weight", type=float, default=0.2)
        
        parser.add_argument("--meta", type=int, default=0)
        parser.add_argument("--meta_lr", type=float, default=0.1) 
        parser.add_argument("--opt_steps", type=int, default=1)
        parser.add_argument("--text", type=str, default="")
        parser.add_argument("--text_weight", type=float, default=0.5)
        parser.add_argument("--run_avg", type=float, default=0.0, help="What fraction of the old encoding to keep.")
        
        parser.add_argument("--run_local", type=int, default=0)
        parser.add_argument("--python_path", type=str, default="python3")
        parser.add_argument("--host", type=str, default="abakus.ddnss.de")
        parser.add_argument("--user", type=str, default="anton")
        parser.add_argument("--mode", type=str, default="stream", choices=["stream", "pic"])
        parser.add_argument('model_args', nargs="*")
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
            old_pid = int(f.read())
        print("Old PID: ", old_pid)
        try:
            os.kill(old_pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        time.sleep(5)
        # check if SIGINT was enough, else kill process
        #try:
        #    os.kill(old_pid, 0)
        #except OSError:
        #    os.kill(old_pid, signal.SIGTERM)
        
        try:
            os.unlink(pidfile)
        except FileNotFoundError:
            print("PID file was already deleted")
            
    if create_new:
        with open(pidfile, 'w') as f:
            f.write(pid)


def clean_pid():
    pidfile = "/tmp/mydaemon.pid"
    os.unlink(pidfile)
    
def clean_folder(path):
    try:
        for f in os.listdir(path):
            os.unlink(os.path.join(path, f))
    except:
        print("Could not clean up", path, "properly...")

