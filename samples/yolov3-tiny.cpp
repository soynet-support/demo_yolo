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


void do_tinyv3(int model_height, int model_width, int input_height, int input_width, vector<string>& params)
{
	vector<string> class_names = {
		//"BG",
		"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
	};

	int nms_count = 300; // 화면에 표시할 최대 객체의 수 <= 모델에서 정의한 최종 객체의 수
	int region_count = 500;
	int batch_size = params.size();
	int engine_serialize = 0;
	char license_file[] = "../mgmt/configs/license_trial.key";
	int device_id = 0;

	int class_count = class_names.size();
	char model_name[] = "yolov3-tiny";

	char engine_file[256];
	sprintf(engine_file, "../mgmt/engines/%s.bin", model_name);
	char weight_file[256];
	sprintf(weight_file, "../mgmt/weights/%s.weights", model_name);

	char cfg_file[256];
	sprintf(cfg_file, "../mgmt/configs/%s.cfg", model_name);
	char log_file[] = "../mgmt/logs/soynet.log";
	char extend_param[1000];

	sprintf(extend_param,
		"BATCH_SIZE=%d ENGINE_SERIALIZE=%d MODEL_NAME=%s CLASS_COUNT=%d NMS_COUNT=%d REGION_COUNT=%d LICENSE_FILE=%s DEVICE_ID=%d ENGINE_FILE=%s WEIGHT_FILE=%s LOG_FILE=%s INPUT_SIZE=%d,%d MODEL_SIZE=%d,%d",
		batch_size, engine_serialize, model_name, class_count, nms_count, region_count, license_file, device_id, engine_file, weight_file, log_file, input_height, input_width, model_height, model_width);
	void* soynet = initSoyNet(cfg_file, extend_param);
	
	vector<uchar> colors(class_count * 3); // 마스크로 사용할 색상 table 저장소
	makeColors(class_count, colors.data(), "bgr"); // mask로 사용할 색상 Table을 미리 만들어 놓는다.
	int thickness = 2, lineType = 8, shift = 0;


	uint64_t dur_microsec = 0;
	uint64_t count = 0;

	string source_type = "video"; //cam or video
	string source_input = params[0];
	//string source_input = "../data/Po720.mp4"; 

	vector<VideoCapture> vcap;
	for (int vidx = 0; vidx < batch_size; vidx++) {
		if (source_type == "video") {
			vcap.emplace_back(VideoCapture(params[vidx]));
			if (!vcap[vidx].isOpened()) {
				printf("Error, Can't open video [%s]\n", source_input.c_str());
				exit(-1);
			}
		}
		else if (source_type == "cam") {
			vcap.emplace_back(VideoCapture(vidx));
			int cam_id = stoi(source_input);
			if (!vcap[vidx].isOpened()) {
				printf("Error, Can't open cam id [%s]", source_input.c_str());
				exit(-1);
			}
		}
		else {
			printf("Error, Not supported source_type [%s], should be [video or cam]\n", source_type.c_str());
			exit(-1);
		}
	}
	int is_break = 0;
	int is_bbox = 1;
	int is_text = 1;
	int is_fps = 1;
	int is_objInfo = 1;

	//Mat resizeMat;
	vector<Mat> img(batch_size);// (input_height, input_width, CV_8UC3);

	int map_size = input_height * input_width * 3;

	while (1) {
		vector<char> input(batch_size * map_size);
		int loop = 1;
		for (int vidx = 0; vidx < batch_size; vidx++) {
			vcap[vidx] >> img[vidx];
			if (img[vidx].empty()) {
				loop = 0;
				break;
			}
			memcpy(&input[vidx*map_size], img[vidx].data, map_size);
		}
		if (loop == 0) {
			break;
		}
		
		vector<BBox> output(batch_size * nms_count);

		uint64_t start_microsec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		feedData(soynet, input.data());
		inference(soynet);
		getOutput(soynet, output.data());
		uint64_t end_microsec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

		uint64_t dur = end_microsec - start_microsec;
		dur_microsec += dur;

		for (int vidx = 0; vidx < batch_size; vidx++) {
			BBox* bbox = &output[vidx*nms_count];
			for (int ridx = 0; ridx < nms_count && bbox[ridx].confid > 0.f; ridx++) {
				if (is_bbox == 1) {//box
					int ori_x1 = bbox[ridx].x1*input_width;
					int ori_y1 = bbox[ridx].y1*input_height;
					int ori_x2 = bbox[ridx].x2*input_width;
					int ori_y2 = bbox[ridx].y2*input_height;
					int class_num = bbox[ridx].obj_id;
					if (is_objInfo == 1) {
						printf("  %3d (%6d %6d %6d %6d) %2d obj=%15s prob=%.6f\n", ridx, ori_x1, ori_y1, ori_x2, ori_y2, class_num, class_names[class_num].c_str(), bbox[ridx].confid);
					}
					int obj_index = bbox[ridx].obj_id;
					Scalar color(colors[obj_index * 3], colors[obj_index * 3 + 1], colors[obj_index * 3 + 2]);
					Rect rect(ori_x1, ori_y1, ori_x2 - ori_x1, ori_y2 - ori_y1);
					rectangle(img[vidx], rect, color, 2, 8, 0);
					if (is_text) {
						Point org(ori_x1, ori_y1 - 3);
						//string prob_s = to_string(roundf(bbox[ridx].prob * 100) / 100).erase(4, 8); // 소수점 3번째 자리에서 반올림 후 string으로 변환
						string text = to_string(ridx) + " " + class_names[class_num] + " " + to_string(bbox[ridx].confid);
						putText(img[vidx], text, org, CV_FONT_HERSHEY_SIMPLEX, 0.5, color);
					}
				}
				if (is_fps == 1) {// fps
					Point org2(15, input_height - 20);
					string text2 = "fps = " + to_string(1000000. / dur);
					putText(img[vidx], text2, org2, CV_FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255));
				}
			}
			String name = "Yolov5" + to_string(vidx);
			imshow(name, img[vidx]);
		}
		printf("%lld time=%.3f milisec fps=%.2f\n", count, dur / 1000., 1000000. / dur);

		int ret = waitKey(1);
		if (ret == 27 || ret == 'q' || ret == 'Q') {
			is_break = 1;
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
		else if (ret == 'f' || ret == 'F') {
			is_fps ^= 1;
		}
		else if (ret == 'i' || ret == 'I') {
			is_objInfo ^= 1;
		}
		if (is_break == 1) break;
		count++;
	}
	printf("count=%lld total_time=%.1f milisec avg_time=%.1f milisec fps=%.2f\n",
		count, dur_microsec / 1000., dur_microsec / 1000. / count, count * 1000000. / dur_microsec);

	freeSoyNet(soynet);
}


int main()
{
	int model_height = 416;
	int model_width = 416;


		string source_type = "video";//cam or video
		string source_input = "NY.mkv";

		VideoCapture vcap;
		if (source_type == "video") {
			if (!vcap.open(source_input)) {
				printf("Error, Can't open video [%s]\n", source_input.c_str());
				exit(-1);
			}
		}
		else if (source_type == "cam") {
			source_input = "0";
			int cam_id = stoi(source_input);
			if (!vcap.open(cam_id)) {
				printf("Error, Can't open cam id [%s]", source_input.c_str());
				exit(-1);
			}
		}
		else {
			printf("Error, Not supported source_type [%s], should be [video or cam]\n", source_type.c_str());
			exit(-1);
		}
		int input_width = vcap.get(3);//640
		int input_height = vcap.get(4);//480

		vector<string> params = { 
			"NY.mkv"
		};

		do_tinyv3(model_height, model_width, input_height, input_width, params);

	return 0;
}
