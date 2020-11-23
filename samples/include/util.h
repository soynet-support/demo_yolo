#pragma once

#include <string>
#include <vector>
#include <random>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

void shuffle(vector<int>& v)
{
	random_device rd;
	mt19937 g(rd());

	shuffle(v.begin(), v.end(), g);
}

void HSV2RGB(float rgb[], const float hsv[])
// ȭ�Ҵ����� ó��
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

void makeColors(int N, unsigned char* colors, const string& mode)
{
	vector<float> fcolors(N * 3);
	vector<int> r(N);
	for (int idx = 0; idx < N; idx++) {
		float hsv[] = { (float)idx / N, 1.f, 1.f };
		float* rgb = fcolors.data() + idx * 3;
		HSV2RGB(rgb, hsv);
		r[idx] = idx;
	}
	shuffle(r);
	if (mode == "rgb" || mode == "RGB") {
		for (int idx = 0; idx < N; idx++) {
			colors[idx * 3 + 0] = (unsigned char)(fcolors[r[idx] * 3 + 0] * 255.f + 0.5f);
			colors[idx * 3 + 1] = (unsigned char)(fcolors[r[idx] * 3 + 1] * 255.f + 0.5f);
			colors[idx * 3 + 2] = (unsigned char)(fcolors[r[idx] * 3 + 2] * 255.f + 0.5f);
		}
	}
	else if (mode == "bgr" || mode == "BGR") {
		for (int idx = 0; idx < N; idx++) {
			colors[idx * 3 + 2] = (unsigned char)(fcolors[r[idx] * 3 + 0] * 255.f + 0.5f);
			colors[idx * 3 + 1] = (unsigned char)(fcolors[r[idx] * 3 + 1] * 255.f + 0.5f);
			colors[idx * 3 + 0] = (unsigned char)(fcolors[r[idx] * 3 + 2] * 255.f + 0.5f);
		}
	}
}
void initVideo(VideoCapture& vcap, int& H, int& W, const string& source_type, const string& source_input)
{
	if (source_type == "video") {
		if (!vcap.open(source_input)) {
		//if (!vcap.open("rtspsrc location=rtsp://192.168.100.21:7001/test.h264 latency=500 ! rtph264depay ! h264parse ! omxh264dec ! appsink max-buffers=1 drop=true sync=false")) {
			printf("Error, Can't open video [%s]\n", source_input.c_str());
			exit(-1);
		}
		printf("Video File [%s] opened!\n", source_input.c_str());
		H = (int)vcap.get(CAP_PROP_FRAME_HEIGHT);
		W = (int)vcap.get(CAP_PROP_FRAME_WIDTH);
	}
	else if (source_type == "cam") {
		int cam_id = stoi(source_input);
		if (!vcap.open(cam_id)) {
			printf("Error, Can't open cam id [%s]", source_input.c_str());
			exit(-1);
		}
		printf("Camera id = [ %d ] open\n", cam_id);
		H = (int)vcap.get(CAP_PROP_FRAME_HEIGHT);
		W = (int)vcap.get(CAP_PROP_FRAME_WIDTH);
	}
	else {
		printf("Error, Not supported source_type [%s], should be [video or cam]\n", source_type.c_str());
		exit(-1);
	}
}
std::string trim_left(const std::string& str)
{
  const std::string pattern = " \f\n\r\t\v";
  return str.substr(str.find_first_not_of(pattern));
}

//
//Right trim
//
std::string trim_right(const std::string& str)
{
  const std::string pattern = " \f\n\r\t\v";
  return str.substr(0,str.find_last_not_of(pattern) + 1);
}

//
//Left and Right trim
//
std::string trim(const std::string& str)
{
  return trim_left(trim_right(str));
}
