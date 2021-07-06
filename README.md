[SoyNet](https://soynet.io/) is an inference optimizing solution for AI models. 
This section describes the process of performing a demo running Yolo (v3-tiny, v3, v4), one of most famous object detection models.

## SoyNet Overview

### Core technology of SoyNet

- Accelerate model inference by maximizing the utilization of numerous cores on the GPU without compromising accuracy (2x to 5x compared to Tensorflow)
- Minimize GPU memory usage (1/5~1/15 level compared to Tensorflow)

### Benefit of SoyNet

- can support customer to  provide AI applications and AI services in time (Time to Market)
- can help application developers to easily execute AI projects without additional technical AI knowledge and experience
- can help customer to reduce H/W (GPU, GPU server) or Cloud Instance cost for the same AI execution (inference)
- can support customer to respond to real-time environments that require very low latency in AI inference

### Features of SoyNet

- Dedicated engine for inference of deep learning models
- Supports NVIDIA and non-NVIDIA GPUs (based on technologies such as CUDA and OpenCL, respectively)
- library files to be easiliy integrated with customer applications
dll file (Windows), so file (Linux) with header or *.lib for building in C/C++

### Folder Structure

```
   ├─mgmt         : SoyNet execution env
   │  ├─configs   : model definitions (*.cfg) and trial license
   │  ├─engines   : SoyNet engine files (it's made at the first time execution.
   │  │             It requires about 30 sec)
   │  ├─logs      : SoyNet log files
   │  └─weights   : weight files for AI models
   └─samples      : folder to build sample demo 
      └─include   : header files
```

### Demo of object detection with Yolo (v3-tiny, v3, v4)

### Prerequisites

### 1.H/W

- GPU : NVIDIA GPU with PASCAL architecture or higher

### 2.S/W

- OS: Ubuntu 18.04LTS
- NVIDIA development environment: CUDA 10.2 / cuDNN 7.6.5 / TensorRT 7.0.0.11
    - For CUDA 10.2, Nvidia-driver 440.33 or higher must be installed
- Others: OpenCV (for reading video files and outputting the screen)

If you have any trouble to make demo environment, you can refer [docker container]([https://github.com/soynet-support/demo_docker](https://github.com/soynet-support/demo_docker), "docker container").

### Run SoyNet Demo

### 1.clone repository

```
$ git clone https://github.com/soynet-support/demo_yolo /demo_yolo
```

### 2.download pre-trained weight files

```
$ cd /demo_yolo/mgmt/weights
$ bash ./download_weights.sh
```

### 3.Demo code Build and Run (C++)

It takes time to create the engine file when it is first executed, and it is loaded immediately after that.

```
$ cd /demo_yolo/samples && make all
```

For yolov3,

```
$ LD_LIBRARY_PATH=/demo_yolo/mgmt:$LD_LIBRARY_PATH ./yolov3
```

### 4.Demo Run (Python)

For yolov3,

```
$ cd /demo_yolo/samples && python3 yolov3.py
```
