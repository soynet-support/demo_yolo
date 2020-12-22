from SoyNet import *
import os, time, sys, cv2, itertools
import numpy as np
from threading import Thread
from queue import Queue

##########################################
NMS_COUNT=300 # be carefull, must be equal to SoyNet.py
##########################################

INPUT = "../../data/NY.mkv"

model_name = "yolov3"

license_file = "../../mgmt/configs/license_trial.key"
device_id = 0
engine_serialize = 0

class_count = len(coco_names)
batch_size = 1
input_height, input_width = (720, 1280)
model_height, model_width = (416, 416)

engine_file = "../../mgmt/engines/%s.bin"%model_name
weight_file = "../../mgmt/weights/%s.weights"%model_name
log_file = "../../mgmt/logs/soynet.log"

cfg_file = "../../mgmt/configs/%s.cfg"%model_name
extend_param = \
	"BATCH_SIZE=%d ENGINE_SERIALIZE=%d MODEL_NAME=%s CLASS_COUNT=%d NMS_COUNT=%d LICENSE_FILE=%s DEVICE_ID=%d ENGINE_FILE=%s WEIGHT_FILE=%s LOG_FILE=%s INPUT_SIZE=%d,%d MODEL_SIZE=%d,%d"%(
	batch_size, engine_serialize, model_name, class_count, NMS_COUNT, license_file, device_id, engine_file, weight_file, log_file, input_height, input_width, model_height, model_width)

handle=initSoyNet(cfg_file, extend_param)

process_count=0
total_time=0
fps=0
avg_fps=0
dur_time=0

PIXEL=[255,220,190,160,150]
COLOR=tuple(itertools.product(PIXEL,PIXEL,PIXEL))
COLOR_NP=np.array(list(itertools.product(PIXEL,PIXEL,PIXEL)))

vcap=cv2.VideoCapture(INPUT)

orig_fps = vcap.get(cv2.CAP_PROP_FPS)
vcap.set(cv2.CAP_PROP_FPS, 15)
new_fps = vcap.get(cv2.CAP_PROP_FPS)
print("orig_fps=%.2f new_fps=%.2f"%(orig_fps, new_fps))

frame_count=int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

def thread_task(handle, input, output, que) :
	begin_time = time.time()
	feedData(handle, input)
	inference(handle)
	getOutput(handle, output)

	end_time = time.time()
	dur_time = end_time-begin_time
	que.put(dur_time)

is_bbox = 1
is_text = 1
is_fps = 1
que=Queue()

img = np.zeros((2, input_height, input_width,3), dtype=np.uint8)
class RIP(Structure): _fields_=[("x1", c_float),("y1", c_float),("x2", c_float),("y2", c_float),("obj_id", c_int),("prob", c_float),]
output_0 = (RIP*NMS_COUNT)()
output_1 = (RIP*NMS_COUNT)()

fidx=0
while(True) :
	cidx = fidx%2 # current idx
	success, img[cidx] = vcap.read()
	if success == False :
		break

	output = output_1 if cidx==1 else output_0

	th=Thread(target=thread_task, args=(handle, img[cidx], output, que))
	th.start()

	# display
	if fidx>0 :
		pidx = (fidx-1)%2 # past idx
		poutput = output_1 if pidx==1 else output_0

		if is_bbox == 1 :
			for nidx in range(NMS_COUNT) :
				r=poutput[nidx]
				#print(r.y1, r.x1, r.y2, r.x2, r.id, r.prob)
				if r.prob == 0. : break
				y1,x1,y2,x2=(int(r.y1*input_height), int(r.x1*input_width) , int(r.y2*input_height), int(r.x2*input_width))
				cv2.rectangle(img[pidx], (x1,y1), (x2,y2), COLOR[r.obj_id], 2 )
				if is_text==1 :
					text = "%s %.5f"%(coco_names[r.obj_id], r.prob)
					cv2.putText(img[pidx], text, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  COLOR[r.obj_id])
				#print("%2d %d %d %d %d %2d %.5f %s"%(nidx, r.x1, r.y1, r.x2, r.y2, r.id, r.prob, coco_names[r.id]))
			#exit()
		if is_fps == 1 :
			text = "fps=%.2f avg_fps=%.2f"%(fps, avg_fps)
			cv2.putText(img[pidx], text, (5, input_height-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 255))

		cv2.imshow("yolo model", img[pidx])

		k = cv2.waitKey(1)

		if k in [27, 'q', 'Q']:
			break
		elif k == ' ':
			cv2.waitKey(0)
		elif k in ['b', 'B']:
			is_bbox = 1 - is_bbox
		elif k in ['t', 'T']:
			print("tttttt")
			is_text = 1 - is_text
		elif k in ['f', 'F']:
			is_fps = 1 - fps

	th.join()
	dur_time = que.get()
	total_time += dur_time
	process_count+=1
	if fidx>0:
		fps=1./dur_time
		avg_fps = process_count / total_time
	fidx +=1

print("total_process_frames=%d"%(process_count))
print("total_process_time=%.3f sec"%(total_time))
print("average_fps=%.2f"%(avg_fps))

# finalize
freeSoyNet(handle)
vcap.release()
cv2.destroyAllWindows()
