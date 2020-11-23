#!/bin/sh
mkdir mgmt/logs mgmt/engines mgmt/weights ./samples/3rdParty
cd mgmt/weights && wget https://soynet.io/demo/yolov4.weights && cd ../../
