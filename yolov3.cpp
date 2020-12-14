#ifdef _WIN32
#pragma comment(lib, "SoyNet.lib")
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <chrono>

#include <thread>
#include <stdlib.h>
#include <stdio.h>


#include "SoyNet.h"
#include "util.h"

using namespace std;
using namespace cv;
using namespace chrono;

#pragma pack(push, 1)
typedef struct { float x1; float y1; float x2; float y2; float confid; int obj_id; } BBox;
#pragma pack(pop)

void task_yolo(const void* soynet, const void* input, void* output, long long* frame_sec)
{
	long long start_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	feedData(soynet, input);
	inference(soynet);
	getOutput(soynet, output);
	long long end_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	frame_sec[0] = int(end_msec - start_msec);
}
void yolov3(int model_height, int model_width, string source_type, const int BatchSize, vector<string>& params, string cfgname)
{
	vector<string> coco_names = {
		//"BG",
		"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
	};
	const int NN = 100; 

	vector<uchar> colors(NN * 3);
	makeColors(NN, colors.data(), "bgr");
	int thickness = 2, lineType = 8, shift = 0;

	long long start_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	void* soynet = initSoyNet(cfgname.c_str(), nullptr);
	long long end_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	long long loading_time = end_msec - start_msec;

	vector<uint8_t> input(BatchSize*model_height*model_width * 3);
	vector<BBox> output_even(BatchSize*NN);
	vector<BBox> output_odd(BatchSize*NN);
	vector<vector<BBox>> output = { output_even, output_odd };

	vector<Mat> orig_even(BatchSize);
	vector<Mat> orig_odd(BatchSize);
	vector<vector<Mat>> origs = { orig_even, orig_odd };
	vector<string> win_names(BatchSize);
	vector<Mat> resizeMats(BatchSize);
	vector<VideoCapture> vcaps(BatchSize);

	int input_height, input_width;
	for (int bidx = 0; bidx < BatchSize; bidx++) {
		initVideo(vcaps[bidx], input_height, input_width, source_type, params[bidx]); 
		win_names[bidx] = "YOLOv3-DarkNet53 : " + to_string(model_height) + " x " + to_string(model_width) + " video " + to_string(bidx);
		namedWindow(win_names[bidx], CV_WINDOW_NORMAL);
		resizeWindow(win_names[bidx], input_width, input_height);
		resizeMats[bidx] = Mat(model_height, model_width, CV_8UC3, input.data() + (bidx*model_height*model_width * 3));
	}

	long long batch_count = 0;
	long long total_msec = 0;
	long long batch_sec = 0;
	int is_bbox = 1;
	int is_text = 1;
	float bps = 0.f;
	bool do_loop = true;
	for (int bidx = 0; bidx < BatchSize; bidx++) {
		do_loop = do_loop && vcaps[bidx].read(origs[batch_count % 2][bidx]);
		if (!do_loop) break;
		resize(origs[batch_count % 2][bidx], resizeMats[bidx], resizeMats[bidx].size(), 0, 0, INTER_LINEAR);
	}
	while (do_loop) {
		thread t1(task_yolo, soynet, (const void*)input.data(), (void*)output[batch_count % 2].data(), &batch_sec);

		if (batch_count > 0) { 
			int id = (batch_count + 1) % 2;
			BBox* bbox_base = output[id].data();

			for (int bidx = 0; bidx < BatchSize; bidx++) {
				Mat& img = origs[id][bidx];
				if (is_bbox) {
					BBox* bbox = bbox_base + bidx*NN;
					for (int idx = 0; idx < NN && bbox[idx].confid>0.f; idx++) {
						Rect rect(bbox[idx].x1, bbox[idx].y1, bbox[idx].x2 - bbox[idx].x1, bbox[idx].y2 - bbox[idx].y1);
						Scalar color(colors[idx * 3], colors[idx * 3 + 1], colors[idx * 3 + 2]);
						rectangle(img, rect, color, thickness, lineType, shift);
						if (is_text) {
							Point org(bbox[idx].x1, bbox[idx].y1 - 3);
							string text = coco_names[bbox[idx].obj_id] + " " + to_string(bbox[idx].confid);
							putText(img, text, org, CV_FONT_HERSHEY_SIMPLEX, 0.5, color);
						}
					}
				}
				string bps_text = string("bps : ") + to_string(bps);
				putText(img, bps_text, Point(5, input_height - 6), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
				imshow(win_names[bidx], img);

				int ret = waitKey(1);
				if (ret == 27 || ret == 'q' || ret == 'Q') {
					do_loop = false;
				}
				else if (ret == ' ') {
					waitKey(0);
				}
				else if (ret == 'b' || ret == 'B') {
					is_bbox ^= 1;
				}
				else if (ret == 't' || ret == 'T') {
					is_text ^= 1;
				}
			}

		}

		t1.join();
		total_msec += batch_sec;
		batch_count++;
		bps = 1000.f / batch_sec;
		printf("%lld %lld[msec] %.2f[bps]\n", batch_count, batch_sec, bps);

		for (int bidx = 0; bidx < BatchSize; bidx++) {
			do_loop = do_loop && vcaps[bidx].read(origs[batch_count % 2][bidx]);
			if (!do_loop) break;
			resize(origs[batch_count % 2][bidx], resizeMats[bidx], resizeMats[bidx].size(), 0, 0, INTER_LINEAR);
		}
	}
	for (int bidx = 0; bidx < BatchSize; bidx++) {
		destroyWindow(win_names[bidx]);
	} 
	printf("loading time = [%lld] msec\n", loading_time);
	printf("batch_count=%lld  %lld[msec] avg=[%.2f]msec  %.2f[bps]\n\n", batch_count, total_msec, total_msec / (float)batch_count, 1000.f / total_msec * batch_count);

	freeSoyNet(soynet);
}

int main(int argc, char** argv)
{
	int model_height = 416;
	int model_width = 416;
	string source_type = "video";
	int batch_size = 1;
	vector<string> params = { "../data/NY.mkv" };
	string config_file = "configs/yolov3.cfg";

	yolov3(model_height, model_width, source_type, batch_size, params, config_file);
}
