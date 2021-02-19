# Master's thesis: "Melanoma analysis with Fractal neural networks"
The goal of the project is to explore the difference in performance between a convolutional neural network and fractal neural network in analysing potential melanoma cases.

## Setting up development environment
Requirements:
- Linux-based machine with GPU
- CUDA 10.x installed
- Docker
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- Git

Installation:
1. Clone the repo
  ```
  gitv clone https://github.com/amyshenin-tech/masters-thesis---melanoma-analysis-with-fnn.git
  ```
2. Start [managing docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/)
3. Build docker image from `./docker/jp.dockerfile`
  ```
  docker build -t <image_name> -f ./docker/jp.dockerfile .
  ```
  make sure, the tag in the `./jupyter.sh` is the same, that you build docker image with.
  
4.  Run `./jupyter.sh`.
5.  If you have docker running on the other machine, be sure to open ssh connection.
    1) Look here, how it [install openssh-server](https://www.youtube.com/watch?v=ur-ctfgzGxs&t=98s&ab_channel=ProgrammingKnowledge) on the computing machine.
    2) Use PuTTY as an ssh client for windows. [Load gateway and open ssh tunel](https://www.skyverge.com/blog/how-to-set-up-an-ssh-tunnel-with-putty/) with port 8888 (jupyter port).
6. Use `docker logs jp` to get the juputer notebook URL (and a tocken).
7. If the notebook, you try to run has time consuming operations, run it in the background mode:
    1) Enter the jp container with `docker exec -it jp bash`.
    2) Run notebook in background `nohup jupyter nbconvert --to notebook --execute mynotebook.ipynb >> output.log &`
8. If you need to download ISIC Archive data:
    1) Enter the jp container with `docker exec -it jp bash`.
    2) Run notebook in background `nohup python download_from_isic.py --limit <number> --offset <number>`
