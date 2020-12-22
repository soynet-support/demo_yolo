# https://stackoverflow.com/questions/22582846/ctypes-structure-arrays-in-python
# http://arkainoh.blogspot.com/2018/08/python.ctypes.html

from ctypes import *
import numpy as np
import numpy.ctypeslib as nc

#lib=cdll.LoadLibrary("../../lib/SoyNet.dll")
lib=cdll.LoadLibrary("../../lib/libSoyNet.so")

lib.initSoyNet.argtypes = [c_char_p, c_char_p]
lib.initSoyNet.restype = c_void_p
def initSoyNet(cfg, extent_params="") :
    if extent_params is None : extent_params=""
    return lib.initSoyNet(cfg.encode("utf8"), extent_params.encode("utf8"))

U8 = nc.ndpointer(dtype=np.uint8, ndim=None, flags='aligned, c_contiguous')
lib.feedData.argtypes=[c_void_p, U8]
lib.feedData.restype=None
def feedData(handle, data) :
    lib.feedData(handle,data)

lib.inference.argtypes=[c_void_p]
lib.inference.restype=None
inference = lib.inference

lib.getOutput.argtypes=[c_void_p, c_void_p]
lib.getOutput.restype=None
def getOutput(handle, output) :
    lib.getOutput(handle, output)
    #lib.getOutput(handle, byref(output))

lib.freeSoyNet.argtypes=[c_void_p]
lib.freeSoyNet.restype=None
freeSoyNet = lib.freeSoyNet

coco_names = [
	#"BG",
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
num_class = len(coco_names)

