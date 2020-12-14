CC=/usr/local/cuda/bin/nvcc
CFLAGS=-Xcompiler -O3 -std=c++11

SRCSDIR=.
PROGDIR=../mgmt

OPENCV=`pkg-config opencv --cflags --libs`
INCS=./include 
LIBS=$(OPENCV) -L../mgmt -lpthread -lSoyNet -lnvidia-ml

all : yolov3-tiny yolov3 yolov4

yolov3-tiny:$(SRCSDIR)/yolov3-tiny.cpp
	$(CC) -o $(PROGDIR)/yolov3-tiny $(SRCSDIR)/yolov3-tiny.cpp $(LIBS) -I$(INCS) $(CFLAGS)

yolov3:$(SRCSDIR)/yolov3.cpp
	$(CC) -o $(PROGDIR)/yolov3 $(SRCSDIR)/yolov3.cpp $(LIBS) -I$(INCS) $(CFLAGS)

yolov4:$(SRCSDIR)/yolov4.cpp
	$(CC) -o $(PROGDIR)/yolov4 $(SRCSDIR)/yolov4.cpp $(LIBS) -I$(INCS) $(CFLAGS)


