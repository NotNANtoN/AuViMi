# AuViMi
*A close-up of my cup with a dragon on it, imagined by deep-daze via my webcam:*

![image](https://user-images.githubusercontent.com/19983153/110973113-1c9a3f00-835d-11eb-85fe-bb4d51c88cd8.png)

*A quick self-portrait, imagined using big-sleep via my webcam:*

![image](https://user-images.githubusercontent.com/19983153/110973359-73a01400-835d-11eb-8507-5297586d9310.png)





AuViMi stands for audio-visual mirror. The idea is to have CLIP generate its interpretation of what your webcam sees, combined with the words thare are spoken.

This implementation assumes that you want to operate on a non-GPU laptop, but have quick connection to a more powerful GPU server.

See it in action (with [deep-daze](https://github.com/lucidrains/deep-daze) as a backbone). You can observe some art, reinterpreted by deep-daze.: 

https://user-images.githubusercontent.com/19983153/110971317-025f6180-835b-11eb-92e2-a5b8faa666a3.mp4

And here's a beautiful self-portrait of NotNANtoN with [big-sleep](https://github.com/lucidrains/big-sleep) as a backbone: 

https://user-images.githubusercontent.com/19983153/110971466-38044a80-835b-11eb-884f-5d52dbd5d06d.mp4

At the moment, we only support the combination of the webcam pictures with a single sentence read from the CLI.

## Usage

**Install**

Install the `requirements.txt` using `python3 -m pip install -r requirements.txt`. Also, install `ffmpeg` on the host server if you want a .mp4 video of the interpretation using `sudo apt-get install ffmpeg`.

**Note**

At the moment, this only works with a remote GPU server that does the computation. Therefore, we assume that ssh is set up. Furthermore, we assume that an ssh-key is used instead of a password to connect to the remote server.

**Commands:**

**You need to set --host, --user, and --python_path!**

`host`could be `university_X.edu.com` and `user` would be your username on that host, e.g. `student_Y`. To find out what to insert for `python_path`, connect to your host and enter `which python3`. This could lead to:

``` python3 app.py --user student_Y --host university_X.edu.com --python_path /usr/bin/python3 ```

Specifying the **operating mode**: If `pic` is set as an operating mode, the user can press `p` to set a new optimization goal - for `stream` the optimization goal is set automatically to the newest pictures from the webcam feed:

``` python3 app.py --mode stream ```

Specifying the backbone, image size (smaller lead to higher FPR but look less nice), batch_size (fewer reduces the amount of VRAM needed on the GPU), whether [meta-learning](https://openai.com/blog/reptile/) should be used, and what meta-learning learning rate is used:

``` python3 app.py --gen_backbone deepdaze --size 256 --batch_size 32 --mode stream --meta 1 --meta_lr 0.2  ```

**Add text** using `--text` and set its weight with `--text_weight`. Setting the weight to `1.0`will ignore the webcam and only visualize the text:

``` python3 app.py --text "A funky human." --text_weight 0.5 ```
