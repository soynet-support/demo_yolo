#!/bin/sh
cd mgmt/weights && \
  wget https://soynet.io/demo/yolov3-tiny.weights && \
  wget https://soynet.io/demo/yolov3.weights && \
  wget https://soynet.io/demo/yolov4.weights && \
  cd ../../
