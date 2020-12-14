AI Deep learning model을 위한 추론전 엔진인 [SoyNet](https://soynet.io, "SOYNET Homepage")을 이용하여
객체감지 모델 중 하나인 Yolov4를 실행하는 데모를 수행하는 과정을 설명한다. 

## SoyNet 개요

### SoyNet의 핵심 기술
 - GPU 상의 수많은 core의 사용율 극대화를 통한 모델 추론 가속 (Tensorflow 대비 2배~5배)
 - GPU 메모리 사용량 최소화 (Tensorflow 대비 1/5~1/15 수준)
 
### SoyNet의 특장점
 - AI deep learning 모델을 이용하여 어플리케이션, 서비스를 하고자 하는 고객에게 최단의 Time-to-Market을 제공
 - 고객사가 자체 보유한 응용 어플리케이션 개발자의 AI 프로젝트 참여를 확대
 - 동일한 AI 실행(추론)을 위해 소요되는 장비 비용의 절감
 - 고도의 Tac-Time을 요구하는 실시간 환경에의 대응 지원
   
### SoyNet 특징
 - Deep Learning 모델의 추론(inference) 전용 엔진 
 - NVIDIA, non-NVIDIA 기반 GPU 지원 (각각 CUDA, OpenCL 등의 기술 기반)
 - 제공형태는 library 파일 
   Windows는 dll, Linux는 so 파일 형태 (개발용 header와 lib는 별도)
 - 폴더 구성
   ```
   ├─mgmt         : SoyNet 실행환경
   │  ├─configs   : 모델정의 파일 (*.cfg)와 임시 라이선스키 포함 
   │  ├─engines   : SoyNet 실행 엔진 파일 생성 (최초 실행 시 1회 생성됨. 30초 가량 소요되며 이후는 생성 파일로 로딩)
   │  ├─logs      : SoyNet log 파일 폴더
   │  └─weights   : 테스트용 모델의 weight 파일포함 (변환 script을 이용하여 SoyNet용으로 변환된 것임)
   └─samples      : 실행 파일을 포함한 빌드를 위한 폴더 
      ├─include   : SoyNet 빌드를 위한 header file 포함 폴더 
      └─3rdParty  : 데모를 위한 3rd Party library 폴더 (OpenCV, CUDA/cuDNN,TensorRT 등등)
   ```
   

## Yolo (v3-tiny, v3, v4)를 이용한 객체 감지 데모 

### 사전 요구사항

#### 1.H/W 
 - GPU : PASCAL 아키텍처 이상의 NVIDIA GPU 

#### 2.S/W
 - 운영체제 : Ubuntu 18.04LTS
 - NVIDIA 개발환경 : CUDA 10.0 / cuDNN 7.6.1 / TensorRT 7.0.0.11
 - 기타 : OpenCV 3.4.5 (영상 파일 읽고 화면 출력하기 위한 용도)
 

### SoyNet 데모 실행

#### 1.Download
```
$ git clone https://github.com/soynet-support/demo_yolo
```

#### 2.create folders & download yolov4 weight file 
```
$ cd demo_yolo
$ bash ./setup.sh
```

<!--
#### 3.download 3rd party libraries 
아래 링크에서 파일을 다운로드하여 samples/3rdParty 폴더에 풀어줌
  - 이미 사전요구사항인 S/W 환경구성이 완료되어 있다면 넘어가도 됨
  - OpenCV의 경우는 데모 프로그램 빌드 시 header, lib 파일이 필요하므로 사전 설치되어 있어야 함

[3rd Party library download](https://drive.google.com/drive/folders/1dF-O_cdstyTt6eKRxN-p8Klv1ITfNHMh?usp=sharing, "3rd party libs")
//-->


#### 3.Demo code Build
```
$ cd samples
$ make all 
```
<!--
$ g++ -std=c++11 -m64 -o ../mgmt/yolov3-tiny ./yolov3-tiny.cpp -I./include -L../mgmt -lSoyNet -lpthread `pkg-config opencv --cflags --libs`
$ g++ -std=c++11 -m64 -o ../mgmt/yolov3 ./yolov3.cpp -I./include -L../mgmt -lSoyNet -lpthread `pkg-config opencv --cflags --libs`
$ g++ -std=c++11 -m64 -o ../mgmt/yolov4 ./yolov4.cpp -I./include -L../mgmt -lSoyNet -lpthread `pkg-config opencv --cflags --libs`
//--> 

#### 4.Demo Code 실행
최초 실행 시 엔진파일 생성에 시간이 소요됨(이후부터는 바로 로딩)
```
# yolo v3
$ LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH ./yolov3
```
