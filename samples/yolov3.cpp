
#include "SoyNet.h"
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <chrono>
#include <thread>
#include <random>
#include <stdlib.h>
#include <stdio.h>
using namespace std;
using namespace cv;
using namespace chrono;

#pragma pack(push, 1)
typedef struct { float x1; float y1; float x2; float y2; int obj_id; float confid; } BBox;
#pragma pack(pop)

static void HSV2RGB(float rgb[], const float hsv[])
{
	if (hsv[1] < FLT_MIN)	rgb[0] = rgb[1] = rgb[2] = hsv[2];
	else {
		const float h = hsv[0];
		const int i = (int)h;
		const float f = h - i;
		const float p = hsv[2] * (1.0f - hsv[1]);

		if (i & 1) {
			const float q = hsv[2] * (1.0f - (hsv[1] * f));
			switch (i) {
			case 1: rgb[0] = q; rgb[1] = hsv[2]; rgb[2] = p; break;
			case 3: rgb[0] = p; rgb[1] = q; rgb[2] = hsv[2]; break;
			default: rgb[0] = hsv[2]; rgb[1] = p; rgb[2] = q;  break;
			}
		}
		else {
			const float t = hsv[2] * (1.0f - (hsv[1] * (1.0f - f)));
			switch (i) {
			case 0: rgb[0] = hsv[2]; rgb[1] = t; rgb[2] = p; break;
			case 2: rgb[0] = p; rgb[1] = hsv[2]; rgb[2] = t; break;
			default: rgb[0] = t; rgb[1] = p; rgb[2] = hsv[2]; break;
			}
		}
	}
}

static void makeColors(int N, unsigned char* colors, const string& mode)
{
	vector<float> fcolors(N * 3);
	vector<int> r(N);
	for (int idx = 0; idx < N; idx++) {
		float hsv[] = { (float)idx / N, 1.f, 1.f };
		float* rgb = fcolors.data() + idx * 3;
		HSV2RGB(rgb, hsv);
		r[idx] = idx;
	}
	{
		random_device rd;
		mt19937 g(rd());
		shuffle(r.begin(), r.end(), g);
	}
	if (mode == "rgb" || mode == "RGB") {
		for (int idx = 0; idx < N; idx++) {
			colors[idx * 3 + 0] = (unsigned char)(fcolors[r[idx] * 3 + 0] * 255.f + 0.5f); colors[idx * 3 + 1] = (unsigned char)(fcolors[r[idx] * 3 + 1] * 255.f + 0.5f); colors[idx * 3 + 2] = (unsigned char)(fcolors[r[idx] * 3 + 2] * 255.f + 0.5f);
		}
	}
	else if (mode == "bgr" || mode == "BGR") {
		for (int idx = 0; idx < N; idx++) {
			colors[idx * 3 + 2] = (unsigned char)(fcolors[r[idx] * 3 + 0] * 255.f + 0.5f); colors[idx * 3 + 1] = (unsigned char)(fcolors[r[idx] * 3 + 1] * 255.f + 0.5f); colors[idx * 3 + 0] = (unsigned char)(fcolors[r[idx] * 3 + 2] * 255.f + 0.5f);
		}
	}
}

void task_yolov3(const void* soynet, const void* input, void* output, long long* frame_sec)
{
	long long start_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	feedData(soynet, input);
	inference(soynet);
	getOutput(soynet, output);
	long long end_msec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	frame_sec[0] = int(end_msec - start_msec);
}
void do_yolov3(int model_height, int model_width, int input_height, int input_width, vector<string>& params)
{
	vector<string> coco_names = {
		//"BG",
		"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
	};


	int nms_count = 50; // 화면에 표시할 최대 객체의 수 <= 모델에서 정의한 최종 객체의 수
	int batch_size = params.size();
	int engine_serialize = 1;
	char license_file[] = "../mgmt/configs/license_trial.key";
	int device_id = 0;

	int class_count = coco_names.size();
	char model_name[] = "yolov3";
	char engine_file[] = "../mgmt/engines/yolov3.bin";
	char weight_file[] = "../mgmt/weights/yolov3.weights";
	char cfg_file[] = "../mgmt/configs/yolov3.cfg";

	char log_file[] = "../mgmt/logs/soynet.log";
	char extend_param[1000];

	sprintf(extend_param,
		"BATCH_SIZE=%d ENGINE_SERIALIZE=%d MODEL_NAME=%s CLASS_COUNT=%d NMS_COUNT=%d LICENSE_FILE=%s DEVICE_ID=%d ENGINE_FILE=%s WEIGHT_FILE=%s LOG_FILE=%s INPUT_SIZE=%d,%d MODEL_SIZE=%d,%d",
		batch_size, engine_serialize, model_name, class_count, nms_count, license_file, device_id, engine_file, weight_file, log_file, input_height, input_width, model_height, model_width);
	void* soynet = initSoyNet(cfg_file, extend_param);

	vector<uchar> colors(class_count * 3); // 마스크로 사용할 색상 table 저장소
	makeColors(class_count, colors.data(), "bgr"); // mask로 사용할 색상 Table을 미리 만들어 놓는다.
	int thickness = 2, lineType = 8, shift = 0;

	// memory를 할당하고, 그걸 이용하는 Mat을 만든다.
	// Mat은 disp와 infer에 동시에 사용되어야 하므로 even, odd를 각각 준비한다.
	vector<uint8_t> origMems(2 * batch_size*input_height*input_width * 3);
	vector<void*> inputs{ origMems.data(), origMems.data() + batch_size * input_height*input_width * 3 };
	vector<vector<Mat>> origMats(2);
	for (int bidx = 0; bidx < batch_size; bidx++) {
		origMats[0].emplace_back(Mat(input_height, input_width, CV_8UC3, origMems.data() + bidx * input_height*input_width * 3));
		origMats[1].emplace_back(Mat(input_height, input_width, CV_8UC3, origMems.data() + (batch_size + bidx)*input_height*input_width * 3));
	}
	vector<BBox> outputMems(2 * batch_size*nms_count);
	vector<void*> outputs{ outputMems.data(), outputMems.data() + batch_size * nms_count };

	vector<string> win_names(batch_size);
	vector<VideoCapture> vcaps(batch_size);
	for (int bidx = 0; bidx < batch_size; bidx++) {
		if (params[bidx].size() == 1) {
			int cam_id = stoi(params[bidx]);
			vcaps[bidx].open(cam_id);
		}
		else {
			vcaps[bidx].open(params[bidx]);
		}

		win_names[bidx] = "YOLOv3 : " + to_string(bidx);
		namedWindow(win_names[bidx], CV_WINDOW_NORMAL);
		resizeWindow(win_names[bidx], input_width, input_height);

		vcaps[bidx].read(origMats[0][bidx]); // even부터 시작, origEvenMem의 memory가 채워진다.
	}

	long long batch_count = 0;
	long long total_msec = 0;
	long long batch_sec = 0;
	int is_bbox = 1;
	int is_text = 1;
	float bps = 0.f;
	bool do_loop = true;
	int eo_flip = 0;

	while (do_loop) {
		thread t1(task_yolov3, soynet, inputs[eo_flip], outputs[eo_flip], &batch_sec); // 입력은 input.data()에 resize되어 들어가 있다.

		if (batch_count > 0) { // Display와 Inference 동작이 병렬로 이루어지는 관계로 최초 실행시에는 Display할 Box정보가 없다.
			BBox* bbox_base = (BBox*)outputs[eo_flip];

			for (int bidx = 0; bidx < batch_size; bidx++) { // batch_size 각각에 대하여 Display한다.
				Mat& img = origMats[eo_flip][bidx]; // 원본 입력 영상
				if (is_bbox) {
					BBox* bbox = bbox_base + bidx * nms_count;
					for (int idx = 0; idx < nms_count && bbox[idx].confid>0.f; idx++) {
						//Rect rect(bbox[idx].x1, bbox[idx].y1, bbox[idx].x2 - bbox[idx].x1, bbox[idx].y2 - bbox[idx].y1);
						int x1 = bbox[idx].x1*input_width;
						int y1 = bbox[idx].y1*input_height;
						int x2 = bbox[idx].x2*input_width;
						int y2 = bbox[idx].y2*input_height;
						int w = x2 - x1;
						int h = y2 - y1;
						Rect rect(x1, y1, w, h);
						Scalar color(colors[idx * 3], colors[idx * 3 + 1], colors[idx * 3 + 2]);

						rectangle(img, rect, color, thickness, lineType, shift);

						if (is_text) {
							Point org(x1, y1 - 3);
							string text = coco_names[bbox[idx].obj_id] + " " + to_string(bbox[idx].confid);
							putText(img, text, org, CV_FONT_HERSHEY_SIMPLEX, 0.5, color);
						}
						//printf("%2d %.3f %.3f %.3f %.3f %2d %15s %.5f\n", bidx, bbox[bidx].x1, bbox[bidx].y1, bbox[bidx].x2, bbox[bidx].y2, bbox[bidx].obj_id, coco_names[bbox[bidx].obj_id].c_str(), bbox[bidx].confid);
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

		for (int bidx = 0; bidx < batch_size; bidx++) {
			do_loop = do_loop && vcaps[bidx].read(origMats[eo_flip][bidx]);
			if (!do_loop) break;
		}
		eo_flip ^= 1;

	}
	for (int bidx = 0; bidx < batch_size; bidx++) {
		destroyWindow(win_names[bidx]);
	}
	printf("batch_count=%lld  %lld[msec] avg=[%.2f]msec  %.2f[bps]\n\n", batch_count, total_msec, total_msec / (float)batch_count, 1000.f / total_msec * batch_count);

	freeSoyNet(soynet);
}


int main()
{
	int model_height = 416;
	int model_width = 416;

	vector<string> params = {
		"NY.mkv"
	};

	int input_height, input_width;
	string source_type = "video";//cam or video
	VideoCapture vcap;
	if (source_type == "video") {
		if (!vcap.open(params[0])) {
			printf("Error, Can't open video [%s]\n", params[0].c_str());
			exit(-1);
		}
		input_width = vcap.get(3);
		input_height = vcap.get(4);
		vcap.release();
	}
	
	else {
		printf("Error, Not supported source_type [%s], should be [video or cam]\n", source_type.c_str());
		exit(-1);
	}

	do_yolov3(model_height, model_width, input_height, input_width, params);
	

	return 0;
}
